#![allow(dead_code)] // TODO: Remove when implementation is complete

use std::{borrow::Cow, sync::OnceLock, time::Duration};

use crate::http::TensorzeroHttpClient;
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
    ProviderInferenceResponseArgs, ProviderInferenceResponseChunk, RequestMessage, Role, Text,
    TextChunk, Usage,
};
use crate::inference::types::{FinishReason, ProviderInferenceResponseStreamInner};
use crate::inference::InferenceProvider;
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};
use crate::tool::{ToolCall, ToolChoice, ToolConfig};

use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};

use crate::inference::TensorZeroEventError;

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

    pub fn model_name(&self) -> &str {
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
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    }
                    .into()
                }))
                .transpose()
            }
            OllamaCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for OllamaProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_url = "http://localhost:11434/api/chat".to_string();
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();

        let request_body =
            serde_json::to_value(OllamaRequest::new(&self.model_name, request.request)).map_err(
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
        )
        .await?;

        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing response: {}", DisplayOrDebugGateway::new(e)),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response_body = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing response: {}", DisplayOrDebugGateway::new(e)),
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
                generic_request: request.request,
            }
            .try_into()?)
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
            ..
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(OllamaRequest::new(&self.model_name, request))
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Ollama request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = "http://localhost:11434/api/chat".to_string();
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
        )
        .await?;
        let stream = stream_ollama(
            PROVIDER_TYPE.to_string(),
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
        )
        .peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
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
    }
    .into()
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
                        let stream_message = data.map(|d| {
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
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: provider_type.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into(),
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
    if let [OllamaContentBlock::Text { text }] = &content.as_slice() {
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

#[derive(Clone, Debug, PartialEq)]
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
    pub r#type: OllamaToolType,
    pub function: OllamaRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for OllamaRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        OllamaRequestToolCall {
            id: &tool_call.id,
            r#type: OllamaToolType::Function,
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
    Tool(OllamaToolRequestMessage<'a>),
}

impl OllamaRequestMessage<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
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
            }
            OllamaRequestMessage::Tool(msg) => msg.content.to_lowercase().contains(value),
        }
    }
}

pub(super) fn prepare_ollama_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
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
        },
    }
}

pub(super) fn tensorzero_to_ollama_message(
    message: &RequestMessage,
) -> Result<Vec<OllamaRequestMessage<'_>>, Error> {
    match message.role {
        Role::User => tensorzero_to_ollama_user_message(&message.content),
        Role::Assistant => tensorzero_to_ollama_assistant_message(&message.content),
    }
}

fn tensorzero_to_ollama_user_message(
    content_blocks: &[ContentBlock],
) -> Result<Vec<OllamaRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the user content blocks

    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                user_content_blocks.push(OllamaContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                messages.push(OllamaRequestMessage::Tool(OllamaToolRequestMessage {
                    content: &tool_result.result,
                    tool_call_id: &tool_result.id,
                }));
            }
            ContentBlock::File(file) => {
                let FileWithPath {
                    file,
                    storage_path: _,
                } = &**file;
                user_content_blocks.push(OllamaContentBlock::ImageUrl {
                    image_url: OllamaImageUrl {
                        // This will only produce an error if we pass in a bad
                        // image with missing data
                        url: format!("data: {}; base64,{}", file.mime_type, file.data()?),
                    },
                });
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
                user_content_blocks.push(OllamaContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        };
    }

    // If there are any user content blocks, combine them into a single user message:
    if !user_content_blocks.is_empty() {
        messages.push(OllamaRequestMessage::User(OllamaUserRequestMessage {
            content: user_content_blocks,
        }));
    }

    Ok(messages)
}

fn tensorzero_to_ollama_assistant_message(
    content_blocks: &[ContentBlock],
) -> Result<Vec<OllamaRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the assistant content blocks

    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                assistant_content_blocks.push(OllamaContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = OllamaRequestToolCall {
                    id: &tool_call.id,
                    r#type: OllamaToolType::Function,
                    function: OllamaRequestFunctionCall {
                        name: &tool_call.name,
                        arguments: &tool_call.arguments,
                    },
                };

                assistant_tool_calls.push(tool_call);
            }
            ContentBlock::ToolResult(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            ContentBlock::File(file) => {
                let FileWithPath {
                    file,
                    storage_path: _,
                } = &**file;
                assistant_content_blocks.push(OllamaContentBlock::ImageUrl {
                    image_url: OllamaImageUrl {
                        // This will only produce an error if we pass in a bad
                        // `Base64Image` (with missing image data)
                        url: format!("data:{};base64,{}", file.mime_type, file.data()?),
                    },
                });
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
                assistant_content_blocks.push(OllamaContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        }
    }

    let content = match assistant_content_blocks.len() {
        0 => None,
        _ => Some(assistant_content_blocks),
    };

    let tool_calls = match assistant_tool_calls.len() {
        0 => None,
        _ => Some(assistant_tool_calls),
    };

    let message = OllamaRequestMessage::Assistant(OllamaAssistantRequestMessage {
        content,
        tool_calls,
    });

    Ok(vec![message])
}

impl OllamaFormat {
    fn ollama_format_from_json_mode(
        json_mode: ModelInferenceRequestJsonMode,
    ) -> Option<OllamaFormat> {
        match json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(OllamaFormat::Json)
            }
            ModelInferenceRequestJsonMode::Off => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OllamaToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OllamaFunction<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OllamaTool<'a> {
    pub(super) r#type: OllamaToolType,
    pub(super) function: OllamaFunction<'a>,
    pub(super) strict: bool,
}

impl<'a> From<&'a ToolConfig> for OllamaTool<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        OllamaTool {
            r#type: OllamaToolType::Function,
            function: OllamaFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum OllamaToolChoice<'a> {
    String(OllamaToolChoiceString),
    Specific(SpecificToolChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OllamaToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SpecificToolChoice<'a> {
    pub(super) r#type: OllamaToolType,
    pub(super) function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct SpecificToolFunction<'a> {
    pub(super) name: &'a str,
}

impl Default for OllamaToolChoice<'_> {
    fn default() -> Self {
        OllamaToolChoice::String(OllamaToolChoiceString::None)
    }
}

impl<'a> From<&'a ToolChoice> for OllamaToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => OllamaToolChoice::String(OllamaToolChoiceString::None),
            ToolChoice::Auto => OllamaToolChoice::String(OllamaToolChoiceString::Auto),
            ToolChoice::Required => OllamaToolChoice::String(OllamaToolChoiceString::Required),
            ToolChoice::Specific(tool_name) => OllamaToolChoice::Specific(SpecificToolChoice {
                r#type: OllamaToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamOption {
    pub(super) include_usage: bool,
}

/// This struct defines the supported paramaeters for the Ollama API
/// See the Ollama API documentation for more details:
/// Se the [Ollama API documentation](https://ollama.readthedocs.io/en/api/#parameters_1)
/// for more details
#[derive(Debug, Serialize)]
pub(super) struct OllamaRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<OllamaRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OllamaFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OllamaFormat {
    Json,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tfs_z: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
}

impl<'a> OllamaRequest<'a> {
    pub fn new(model_name: &'a str, request: &'a ModelInferenceRequest<'a>) -> Self {
        let messages = tensorzero_to_ollama_messages(&request.messages);

        OllamaRequest {
            model: model_name,
            messages,
            tools: None, // TODO: Add tool support
            format: OllamaFormat::ollama_format_from_json_mode(request.json_mode),
            options: None, // TODO: Add options support
            stream: false,
            keep_alive: None,
        }
    }
}

fn tensorzero_to_ollama_messages<'a>(
    _messages: &'a [RequestMessage],
) -> Vec<OllamaRequestMessage<'a>> {
    // For now, return empty vector - this is a stub implementation
    // TODO: Implement proper message conversion
    vec![]
}

#[derive(Debug, Deserialize)]
struct OllamaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaResponseMessage,
    #[serde(rename = "eval_count")]
    completion_tokens: Option<u32>,
    #[serde(rename = "prompt_eval_count")]
    prompt_tokens: Option<u32>,
    done: bool,
}

// Streaming response structures
#[derive(Debug, Deserialize, Serialize)]
struct OllamaChatChunkMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OllamaChatChunk {
    model: String,
    created_at: String,
    message: OllamaChatChunkMessage,
    done: bool,
    #[serde(rename = "eval_count")]
    completion_tokens: Option<u32>,
    #[serde(rename = "prompt_eval_count")]
    prompt_tokens: Option<u32>,
    #[serde(rename = "total_duration")]
    total_duration: Option<u64>,
    #[serde(rename = "load_duration")]
    load_duration: Option<u64>,
    #[serde(rename = "prompt_eval_duration")]
    prompt_eval_duration: Option<u64>,
    #[serde(rename = "eval_duration")]
    eval_duration: Option<u64>,
}

struct OllamaResponseWithMetadata<'a> {
    response: OllamaResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<OllamaResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;

    fn try_from(value: OllamaResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OllamaResponseWithMetadata {
            response,
            raw_response,
            latency,
            raw_request,
            generic_request,
        } = value;

        // Convert Ollama response to TensorZero format
        let content = vec![ContentBlockOutput::Text(Text {
            text: response.message.content,
        })];

        let usage = Usage {
            input_tokens: response.prompt_tokens.unwrap_or(0),
            output_tokens: response.completion_tokens.unwrap_or(0),
        };

        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();

        // Ollama doesn't provide explicit finish reasons, so we default to Stop
        let finish_reason = if response.done {
            FinishReason::Stop
        } else {
            FinishReason::Unknown
        };

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                latency,
                finish_reason: Some(finish_reason),
            },
        ))
    }
}

/// Maps an Ollama chunk to a TensorZero chunk for streaming inferences
fn ollama_to_tensorzero_chunk(
    chunk: OllamaChatChunk,
    latency: Duration,
    _tool_call_ids: &mut [String], // Ollama doesn't support tool calls in streaming yet
) -> ProviderInferenceResponseChunk {
    // Serialize the chunk first before we move any values out of it
    let raw_chunk = serde_json::to_string(&chunk).unwrap_or_default();

    let mut content = vec![];
    let mut finish_reason = None;

    // Ollama provides content in the message.content field
    if !chunk.message.content.is_empty() {
        content.push(ContentBlockChunk::Text(TextChunk {
            text: chunk.message.content,
            id: "0".to_string(),
        }));
    }

    // Check if the stream is done
    if chunk.done {
        finish_reason = Some(FinishReason::Stop);
    }

    // Convert usage if available
    let usage =
        if chunk.done && (chunk.completion_tokens.is_some() || chunk.prompt_tokens.is_some()) {
            Some(Usage {
                input_tokens: chunk.prompt_tokens.unwrap_or(0),
                output_tokens: chunk.completion_tokens.unwrap_or(0),
            })
        } else {
            None
        };

    ProviderInferenceResponseChunk::new(content, usage, raw_chunk, latency, finish_reason)
}
