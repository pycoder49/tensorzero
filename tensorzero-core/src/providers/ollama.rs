use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Duration;
use minijinja::filters::default;
use tokio::time::Instant;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{warn_discarded_thought_block, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::resolved_input::FileWithPath;
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, ContentBlockChunk,
    ContentBlockOutput, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, RequestMessage, Role, Text, TextChunk, Usage,
};
use crate::inference::types::{
    FinishReason, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner,
};
use crate::inference::{InferenceProvider, TensorZeroEventError};
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};
use crate::providers::groq::{GroqContentBlock, GroqRequestMessage, GroqRequestToolCall};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("OLLAMA_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "Ollama";
pub const PROVIDER_TYPE: &str = "ollama";

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct OllamaProvider {
    model_name: String,
    #[serde(skip)]
    credentials: OllamaCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<OllamaCredentials> = OnceLock::new();

impl OllamaProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(OllamaProvider {
            model_name,
            credentials,
        })
    }

    pub fn model_name(&self) -> &str{
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum OllamaCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for OllamaCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OllamaCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OllamaCredentials::Dynamic(key_name)),
            Credential::None => Ok(OllamaCredentials::None),
            Credential::Missing => Ok(OllamaCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Ollama provider".to_string(),
            })),
        }
    }
}

impl OllamaCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            OllamaCredentials::Static(api_key) => Ok(Some(api_key)),
            OllamaCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }.into()
                })).transpose()
            }
            OllamaCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for OllamaProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_url = "http://localhost:11434/api/generate".to_string();
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();

        let request_body =
            serde_json::to_value(OllamaRequest::new(&self.model_name, request.request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Ollama request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                },
            )?;

        let mut request_builder = http_client.post(request_url);
        if let Some(key) = api_key {
            request_builder = request_builder.bearer_auth(key.expose_secret());
        }

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.request.extra_body,
            &request.request.extra_headers,
            model_provider,
            request.model_name,
            request_body,
            request_builder,
        ).await?;

        let latency = Latency::NonStreaming{
            response_time: start_time.elapsed(),
        };

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InfereceServer {
                    message: format!(
                        "Error parsing response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response_body = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            Ok(OllamaResponseWithMetadata {
                response: response_body,
                latency,
                raw_response,
                raw_request: raw_request.clone(),
                generic_request: request,
            }.try_into()?)
        } else {
            Err(handle_ollama_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(GroupRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Ollama request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = "http://localhost:11434/api/generate".to_string();
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client.post(request_url);
        if let Some(key) = api_key {
            request_builder = request_builder.bearer_auth(key.expose_secret());
        }
        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        ).await?;
        let stream = stream_ollama(
            PROVIDER_TYPE.to_string(),
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
        ).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string();
        }.into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string();
        }.into())
    }
}

pub async fn convert_stream_error(provider_type: String, e: reqwest_eventsource::Error) -> Error {
    let message = e.to_string();
    let mut raw_response = None;
    if let reqwest_eventsource::Error::InvalidStatusCode(_, resp) = e {
        raw_response = resp.text().await.ok();
    }
    ErrorDetails::InferenceServer {
        message,
        raw_request: None,
        raw_response,
        provider_type,
    }.into()
}

pub fn stream_ollama(
    provider_type: String,
    event_source: impl Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    let mut tool_call_ids = Vec::new();
    Box::pin(async_stream::stream! {
        futures::pin_mut!(event_source);
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    match e {
                        TensorZeroEventError::TensorZero(e) => {
                            yield Err(e);
                        }
                        TensorZeroEventError::EventSource(e) => {
                            yield Err(convert_stream_error(provider_type.clone(), e).await);
                        }
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<OllamaChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing Ollama chunk. Error: {e}",
                                    ),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: provider_type.clone(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            ollama_to_tensorzero_chunk(d, latency, &mut tool_call_ids)
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

pub(super) fn handle_ollama_error(
    response_code: StatusCode,
    response_body: &str,
    provider_type: &str,
) -> Error {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            message: response_body.to_string(),
            raw_request: None,
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OllamaSystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OllamaUserRequestMessage<'a> {
    #[serde(serialize_with = "serialize_text_content_vec")]
    pub(super) content: Vec<OllamaContentBlock<'a>>,
}

fn serialize_text_content_vec<S>(
    content: &Vec<OllamaContentBlock<'_>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If we have a single text block, serialize it as a string
    // to stay compatible with older providers which may not support content blocks
    if let[OllamaContentBlock::Text{ text }] = &content.as_slice() {
        text.serialize(serializer)
    } else {
        content.serialize(serializer)
    }
}

fn serialize_optional_text_content_vec<S>(
    content: &Option<Vec<OllamaContentBlock<'_>>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match content {
        Some(vec) => serialize_text_content_vec(vec, serializer),
        None => serializer.serialize_none(),
    }
}

#[derive(Clone, Debug, PartialEw)]
pub enum OllamaContentBlock<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: OllamaImageUrl },
    Unknown { data: Cow<'a, Value> },
}

impl Serialize for OllamaContentBlock<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Helper<'a> {
            Text { text: &'a str },
            ImageUrl { image_url: &'a OllamaImageUrl },
        }
        match self {
            OllamaContentBlock::Text { text } => Helper::Text { text }.serialize(serializer),
            OllamaContentBlock::ImageUrl { image_url } => {
                Helper::ImageUrl { image_url }.serialize(serializer)
            }
            OllamaContentBlock::Unknown { data } => data.serialize(serializer),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OllamaImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OllamaRequestFunctionCall<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct OllamaRequestToolCall<'a> {
    pub id: &'a str,
    pub retype: OllamaToolType,
    pub function: OllamaRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for OllamaRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        OllamaRequestToolCall {
            id: &tool_call.id,
            retype: OllamaToolType::Function,
            function: OllamaRequestFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OllamaAssistantRequestMessage<'a> {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_text_content_vec"
    )]
    pub content: Option<Vec<OllamaContentBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OllamaToolRequestMessage<'a> {
    pub content: &'a str,
    pub tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum OllamaRequestMessage<'a> {
    System(OllamaSystemRequestMessage<'a>),
    User(OllamaUserRequestMessage<'a>),
    Assistant(OllamaAssistantRequestMessage<'a>),
    Tool(OllameToolRequestMessage<'a>),
}

impl OllamaRequestMessage<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self{
            OllamaRequestMessage::System(msg) => msg.content.to_lowercase().contains(value),
            OllamaRequestMessage::User(msg) => msg.content.iter().any(|c| match c {
                OllamaContentBlock::Text { text } => text.to_lowercase().contains(value),
                OllamaContentBlock::ImageUrl { .. } => false,
                // Don't inspect the contents of 'unknown' blocks
                OllamaContentBlock::Unknown { data: _ } => false,
            }),
            OllamaRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    content.iter().any(|c| match c {
                        OllamaContentBlock::Text { text } => text.to_lowercase().contains(value),
                        OllamaContentBlock::ImageUrl { .. } => false,
                        OllamaContentBlock::Unknown { data: _ } => false,
                    })
                } else {
                    false
                }
            },
            OllamaRequestMessage::Tool(msg) => msg.content.to_lowercase().contains(value),
        }
    }
}

pub(super) fn prepare_ollama_messages<'a>(
    request: &'a ModelInferenceRequest<'_>
) -> Result<Vec<OllamaRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        messages.extend(tensorzero_to_ollama_message(message)?);
    }
    if let Some(system_msg) =
        tensorzero_to_ollama_system_message(request.system.as_deref(), request.json_mode, &messages)
    {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise, convert the tool choice and tools to Ollama format
pub(super) fn prepare_ollama_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> (
    Option<Vec<OllamaTool<'a>>>,
    Option<OllamaToolChoice<'a>>,
    Option<bool>,
) {
    match &request.tool_config {
        None => (None, None, None),
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None, None);
            }
            let tools = Some(tool_config.tools_available.iter().map(Into::into).collect());
            let tool_choice = Some((&tool_config.tool_choice).into());
            let parallel_tool_calls = tool_config.parallel_tool_calls;
            (tools, tool_choice, parallel_tool_calls)
        }
    }
}

/// If ModelInferenceRequestJsonMode::On and the system message or instructions do not contain "JSON"
/// the request will return an error
/// So, we need to format the instructions to include "Response using JSON." if it doesn't already
pub(super) fn tensorzero_to_ollama_system_message<'a>(
    system: Option<&'a str>,
    json_mode: ModelInferenceRequestJsonMode,
    messages: &[OllamaRequestMessage<'a>],
) -> Option<OllamaRequestMessage<'a>> {
    match system {
        Some(system) => {
            match json_mode {
                ModelInferenceRequestJsonMode::On => {
                    if messages
                        .iter()
                        .any(|msg| msg.content_contains_case_insensitive("json"))
                        || system.to_lowercase().contains("json")
                    {
                        OllamaRequestMessage::System(OllamaSystemRequestMessage {
                            content: Cow::Borrowed(system),
                        })
                    } else {
                        let formatted_instructions = format!("Respond using JSON.\n\n{system}");
                        OllamaRequestMessage::System(OllamaSystemRequestMessage {
                            content: Cow::Owned(formatted_instructions),
                        })
                    }
                }

                // If JSON mode is either off or strict, we don't need to do anything special
                _ => OllamaRequestMessage::System(OllamaSystemRequestMessage {
                    content: Cow::Borrowed(system),
                }),
            }
            .into()
        }
        None => match json_mode {
            ModelInferenceRequestJsonMode::On => {
                Some(OllamaRequestMessage::System(OllamaSystemRequestMessage {
                    content: Cow::Owned("Respond using JSON.".to_string()),
                }))
            }
            _ => None,
        }
    }
}

pub(super) fn tensorzero_to_ollama_message<'a>(
    message: &RequestMessage,
) -> Result<Vec<OllamaRequestMessage<'_>>, Error> {
    match message.role {
        Role::User => tensorzero_to_ollama_user_message(&message.content),
        Role::Assistant => tensorzero_to_ollama_assistant_message(&message.content),
    }
}


