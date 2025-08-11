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
}
