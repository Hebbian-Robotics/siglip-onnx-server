//! HTTP contract and ONNX Runtime adapter for text-only `SigLIP 2` embedding.

use std::mem::size_of;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result, bail, ensure};
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderValue, StatusCode, header::RETRY_AFTER};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use ort::session::Session;
use ort::session::builder::PrepackedWeights;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::info;

use crate::config::ServiceConfig;

/// Builds the text-only public API. Unsupported modalities have no route, so
/// this CPU service cannot be mistaken for an indexing embedder.
pub fn build_router(text_embedder: Arc<OnnxTextEmbedder>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/embeddings", post(openai_embed))
        .with_state(text_embedder)
}

/// Loaded tokenizer and a bounded pool of ONNX text sessions.
pub struct OnnxTextEmbedder {
    config: ServiceConfig,
    tokenizer: Tokenizer,
    pad_token_id: u32,
    session_pool: Arc<OnnxSessionPool>,
    admission_controller: InferenceAdmissionController,
}

/// Owns independent mutable sessions and returns each one to the pool after a
/// request, including when inference exits early with an error.
struct OnnxSessionPool {
    available_sessions: Mutex<Vec<Session>>,
    session_available: Condvar,
}

struct OnnxSessionLease {
    session: Option<Session>,
    session_pool: Arc<OnnxSessionPool>,
}

/// Limits active plus waiting requests before they occupy blocking executor
/// threads. Saturation is an expected HTTP outcome rather than hidden queuing.
struct InferenceAdmissionController {
    available_request_slots: Arc<Semaphore>,
}

enum InferenceAdmission {
    Admitted(InferenceAdmissionPermit),
    Saturated,
}

struct InferenceAdmissionPermit {
    _request_slot: OwnedSemaphorePermit,
}

impl OnnxSessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        debug_assert!(!sessions.is_empty());
        Self {
            available_sessions: Mutex::new(sessions),
            session_available: Condvar::new(),
        }
    }

    fn checkout(self: &Arc<Self>) -> Result<OnnxSessionLease> {
        let mut available_sessions = self
            .available_sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session pool mutex was poisoned"))?;
        while available_sessions.is_empty() {
            available_sessions = self
                .session_available
                .wait(available_sessions)
                .map_err(|_| anyhow::anyhow!("ONNX session pool mutex was poisoned"))?;
        }
        let session = available_sessions
            .pop()
            .context("ONNX session pool signaled availability without a session")?;
        Ok(OnnxSessionLease {
            session: Some(session),
            session_pool: Arc::clone(self),
        })
    }
}

impl OnnxSessionLease {
    fn session_mut(&mut self) -> Result<&mut Session> {
        self.session
            .as_mut()
            .context("ONNX session lease no longer owns a session")
    }
}

impl Drop for OnnxSessionLease {
    fn drop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        // A poisoned availability lock does not invalidate the ONNX session.
        // Recovering the guard here preserves pool capacity after an unrelated
        // panic and avoids silently leaking a multi-gigabyte session.
        let mut available_sessions = self
            .session_pool
            .available_sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        available_sessions.push(session);
        self.session_pool.session_available.notify_one();
    }
}

impl InferenceAdmissionController {
    fn new(session_count: usize, maximum_pending_requests: usize) -> Result<Self> {
        let maximum_admitted_requests = session_count
            .checked_add(maximum_pending_requests)
            .context("session and pending-request counts overflow")?;
        Ok(Self {
            available_request_slots: Arc::new(Semaphore::new(maximum_admitted_requests)),
        })
    }

    fn try_admit(&self) -> InferenceAdmission {
        match Arc::clone(&self.available_request_slots).try_acquire_owned() {
            Ok(request_slot) => InferenceAdmission::Admitted(InferenceAdmissionPermit {
                _request_slot: request_slot,
            }),
            Err(_) => InferenceAdmission::Saturated,
        }
    }
}

impl OnnxTextEmbedder {
    /// Loads immutable, local artifacts and validates the exported graph shape
    /// before accepting network traffic.
    ///
    /// # Errors
    ///
    /// Returns an error when the native runtime, tokenizer, or ONNX graph
    /// cannot be loaded, or when the artifact contract does not match `SigLIP 2`.
    pub fn load(config: ServiceConfig) -> Result<Self> {
        info!("initializing ONNX Runtime");
        let onnx_runtime_environment =
            ort::init_from(config.onnx_runtime_library_path()).map_err(|error| {
                anyhow::anyhow!("loading the configured ONNX Runtime shared library: {error}")
            })?;
        ensure!(
            onnx_runtime_environment.commit(),
            "ONNX Runtime was already initialized before the SigLIP 2 service loaded its configured library"
        );

        info!("loading pinned SigLIP 2 tokenizer");
        let mut tokenizer = Tokenizer::from_file(config.tokenizer_path()).map_err(|error| {
            anyhow::anyhow!(
                "loading SigLIP 2 tokenizer from {}: {error}",
                config.tokenizer_path().display(),
            )
        })?;
        let pad_token_id =
            configure_fixed_length_tokenizer(&mut tokenizer, config.text_max_token_length())?;

        let shared_prepacked_weights = PrepackedWeights::new();
        let mut sessions = Vec::with_capacity(config.session_count());
        for session_index in 0..config.session_count() {
            info!(
                session_number = session_index + 1,
                session_count = config.session_count(),
                intra_operation_threads = config.intra_operation_threads(),
                "creating SigLIP 2 ONNX text session"
            );
            let session = load_onnx_session(&config, &shared_prepacked_weights)?;
            if session_index == 0 {
                validate_graph_contract(&session, &config)?;
            }
            sessions.push(session);
        }

        info!("running SigLIP 2 ONNX startup probe");
        let admission_controller = InferenceAdmissionController::new(
            config.session_count(),
            config.maximum_pending_requests(),
        )?;
        let text_embedder = Self {
            config,
            tokenizer,
            pad_token_id,
            session_pool: Arc::new(OnnxSessionPool::new(sessions)),
            admission_controller,
        };
        let startup_probe = TextEmbeddingBatch::from_texts(
            vec!["siglip startup contract probe".to_owned()],
            text_embedder.config.maximum_batch_size(),
        )?;
        let startup_admission = match text_embedder.admission_controller.try_admit() {
            InferenceAdmission::Admitted(admission_permit) => admission_permit,
            InferenceAdmission::Saturated => {
                bail!("new ONNX service was saturated before its startup probe")
            }
        };
        text_embedder.embed_text_batch(startup_probe, startup_admission)?;
        Ok(text_embedder)
    }

    /// Embeds one validated, ordered batch after fixed-length tokenization.
    ///
    /// # Errors
    ///
    /// Returns an error for tokenizer or runtime failures, and an ONNX result
    /// that violates the vector contract.
    fn embed_text_batch(
        &self,
        text_batch: TextEmbeddingBatch,
        _admission_permit: InferenceAdmissionPermit,
    ) -> std::result::Result<TextEmbeddingBatchOutput, TextEmbeddingOperationError> {
        let prepared_batch = self.prepare_text_embedding_batch(text_batch)?;
        let embeddings = self.run_prepared_text_embedding_batch(&prepared_batch)?;
        Ok(TextEmbeddingBatchOutput {
            embeddings,
            input_token_count: prepared_batch.input_token_count,
        })
    }

    fn prepare_text_embedding_batch(
        &self,
        text_batch: TextEmbeddingBatch,
    ) -> std::result::Result<PreparedTextEmbeddingBatch, TextEmbeddingOperationError> {
        match text_batch {
            TextEmbeddingBatch::Texts(texts) => {
                let text_references: Vec<&str> = texts.iter().map(String::as_str).collect();
                let encodings = self
                    .tokenizer
                    .encode_batch(text_references, true)
                    .map_err(|error| anyhow::anyhow!("tokenizing text batch: {error}"))?;
                let input_ids = flatten_token_ids(&encodings, self.config.text_max_token_length())?;
                let attention_mask =
                    flatten_attention_masks(&encodings, self.config.text_max_token_length())?;
                let input_token_count = count_unpadded_tokens(&encodings);
                Ok(PreparedTextEmbeddingBatch {
                    batch_size: texts.len(),
                    input_ids,
                    attention_mask,
                    input_token_count,
                })
            }
            TextEmbeddingBatch::TokenIds(token_sequences) => prepare_token_id_batch(
                &self.tokenizer,
                token_sequences,
                self.config.text_max_token_length(),
                self.pad_token_id,
            )
            .map_err(TextEmbeddingOperationError::Rejected),
        }
    }

    fn run_prepared_text_embedding_batch(
        &self,
        prepared_batch: &PreparedTextEmbeddingBatch,
    ) -> Result<Vec<Vec<f32>>> {
        let input_ids_tensor = Tensor::from_array((
            [
                prepared_batch.batch_size,
                self.config.text_max_token_length(),
            ],
            prepared_batch.input_ids.clone(),
        ))
        .map_err(|error| anyhow::anyhow!("building ONNX input_ids tensor: {error}"))?;
        let attention_mask_tensor = Tensor::from_array((
            [
                prepared_batch.batch_size,
                self.config.text_max_token_length(),
            ],
            prepared_batch.attention_mask.clone(),
        ))
        .map_err(|error| anyhow::anyhow!("building ONNX attention_mask tensor: {error}"))?;

        let mut session_lease = self.session_pool.checkout()?;
        let outputs = session_lease
            .session_mut()?
            .run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            })
            .map_err(|error| anyhow::anyhow!("running SigLIP 2 text ONNX graph: {error}"))?;
        let embedding_output = outputs.get(self.config.output_name()).with_context(|| {
            format!(
                "ONNX graph did not return configured output {:?}",
                self.config.output_name()
            )
        })?;
        let embedding_array = embedding_output
            .try_extract_array::<f32>()
            .map_err(|error| {
                anyhow::anyhow!("extracting f32 embeddings from ONNX output: {error}")
            })?;

        normalize_embedding_batch(
            embedding_array
                .as_slice()
                .context("ONNX output must be contiguous")?,
            prepared_batch.batch_size,
            self.config.embedding_dimension(),
        )
    }
}

fn load_onnx_session(
    config: &ServiceConfig,
    shared_prepacked_weights: &PrepackedWeights,
) -> Result<Session> {
    Session::builder()
        .map_err(|error| anyhow::anyhow!("creating ONNX Runtime session builder: {error}"))?
        .with_prepacked_weights(shared_prepacked_weights)
        .map_err(|error| anyhow::anyhow!("sharing prepacked ONNX weights: {error}"))?
        .with_intra_threads(config.intra_operation_threads())
        .map_err(|error| {
            anyhow::anyhow!("configuring ONNX Runtime intra-operation threads: {error}")
        })?
        .with_inter_threads(1)
        .map_err(|error| {
            anyhow::anyhow!("configuring ONNX Runtime inter-operation threads: {error}")
        })?
        .commit_from_file(config.model_path())
        .map_err(|error| {
            anyhow::anyhow!(
                "loading SigLIP 2 text ONNX graph from {}: {error}",
                config.model_path().display()
            )
        })
}

fn configure_fixed_length_tokenizer(tokenizer: &mut Tokenizer, token_length: usize) -> Result<u32> {
    let pad_token = "<pad>";
    let pad_token_id = tokenizer.token_to_id(pad_token).with_context(|| {
        format!("SigLIP 2 tokenizer does not define required pad token {pad_token:?}")
    })?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            direction: TruncationDirection::Right,
            max_length: token_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .map_err(|error| {
            anyhow::anyhow!("configuring fixed SigLIP 2 tokenizer truncation: {error}")
        })?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(token_length),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: pad_token_id,
        pad_type_id: 0,
        pad_token: pad_token.to_owned(),
    }));
    Ok(pad_token_id)
}

fn validate_graph_contract(session: &Session, config: &ServiceConfig) -> Result<()> {
    let input_names: Vec<&str> = session
        .inputs()
        .iter()
        .map(ort::value::Outlet::name)
        .collect();
    for required_input_name in ["input_ids", "attention_mask"] {
        ensure!(
            input_names.contains(&required_input_name),
            "SigLIP 2 text ONNX graph is missing required input {required_input_name:?}; found {input_names:?}"
        );
    }

    let output_names: Vec<&str> = session
        .outputs()
        .iter()
        .map(ort::value::Outlet::name)
        .collect();
    ensure!(
        output_names.contains(&config.output_name()),
        "SigLIP 2 text ONNX graph is missing configured output {:?}; found {output_names:?}",
        config.output_name()
    );
    Ok(())
}

fn flatten_token_ids(encodings: &[tokenizers::Encoding], token_length: usize) -> Result<Vec<i64>> {
    flatten_encoding_values(
        encodings,
        token_length,
        tokenizers::Encoding::get_ids,
        "token IDs",
    )
}

fn flatten_attention_masks(
    encodings: &[tokenizers::Encoding],
    token_length: usize,
) -> Result<Vec<i64>> {
    flatten_encoding_values(
        encodings,
        token_length,
        tokenizers::Encoding::get_attention_mask,
        "attention masks",
    )
}

fn flatten_encoding_values(
    encodings: &[tokenizers::Encoding],
    token_length: usize,
    select_values: impl Fn(&tokenizers::Encoding) -> &[u32],
    label: &str,
) -> Result<Vec<i64>> {
    let mut flattened_values = Vec::with_capacity(encodings.len() * token_length);
    for (encoding_index, encoding) in encodings.iter().enumerate() {
        let values = select_values(encoding);
        ensure!(
            values.len() == token_length,
            "SigLIP 2 tokenizer produced {} {label} for input {encoding_index}, expected {token_length}",
            values.len()
        );
        flattened_values.extend(values.iter().map(|value| i64::from(*value)));
    }
    Ok(flattened_values)
}

fn count_unpadded_tokens(encodings: &[tokenizers::Encoding]) -> usize {
    encodings
        .iter()
        .flat_map(tokenizers::Encoding::get_attention_mask)
        .filter(|attention_value| **attention_value != 0)
        .count()
}

fn prepare_token_id_batch(
    tokenizer: &Tokenizer,
    token_sequences: Vec<Vec<u32>>,
    token_length: usize,
    pad_token_id: u32,
) -> std::result::Result<PreparedTextEmbeddingBatch, TextEmbeddingRequestRejection> {
    let batch_size = token_sequences.len();
    let mut input_ids = Vec::with_capacity(batch_size * token_length);
    let mut attention_mask = Vec::with_capacity(batch_size * token_length);
    let mut input_token_count = 0_usize;

    for (input_index, token_sequence) in token_sequences.into_iter().enumerate() {
        for token_id in &token_sequence {
            if tokenizer.id_to_token(*token_id).is_none() {
                return Err(TextEmbeddingRequestRejection::UnknownTokenId {
                    input_index,
                    token_id: *token_id,
                });
            }
        }

        let retained_token_count = token_sequence.len().min(token_length);
        input_ids.extend(
            token_sequence
                .into_iter()
                .take(retained_token_count)
                .map(i64::from),
        );
        attention_mask.extend(std::iter::repeat_n(1_i64, retained_token_count));

        let padding_token_count = token_length - retained_token_count;
        input_ids.extend(std::iter::repeat_n(
            i64::from(pad_token_id),
            padding_token_count,
        ));
        attention_mask.extend(std::iter::repeat_n(0_i64, padding_token_count));
        input_token_count += retained_token_count;
    }

    Ok(PreparedTextEmbeddingBatch {
        batch_size,
        input_ids,
        attention_mask,
        input_token_count,
    })
}

struct PreparedTextEmbeddingBatch {
    batch_size: usize,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    input_token_count: usize,
}

struct TextEmbeddingBatchOutput {
    embeddings: Vec<Vec<f32>>,
    input_token_count: usize,
}

#[derive(Debug, Error)]
enum TextEmbeddingOperationError {
    #[error(transparent)]
    Rejected(#[from] TextEmbeddingRequestRejection),
    #[error(transparent)]
    Inference(#[from] anyhow::Error),
}

fn normalize_embedding_batch(
    flat_embeddings: &[f32],
    expected_count: usize,
    embedding_dimension: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected_values = expected_count
        .checked_mul(embedding_dimension)
        .context("embedding count and dimension overflow")?;
    ensure!(
        flat_embeddings.len() == expected_values,
        "ONNX output has {} values, expected {expected_count} vectors of dimension {embedding_dimension}",
        flat_embeddings.len()
    );

    flat_embeddings
        .chunks_exact(embedding_dimension)
        .enumerate()
        .map(|(embedding_index, embedding)| normalize_embedding(embedding, embedding_index))
        .collect()
}

fn normalize_embedding(embedding: &[f32], embedding_index: usize) -> Result<Vec<f32>> {
    let squared_norm = embedding.iter().try_fold(0.0_f32, |total, component| {
        if !component.is_finite() {
            bail!("ONNX output embedding {embedding_index} contains a non-finite component");
        }
        Ok(total + component * component)
    })?;
    let norm = squared_norm.sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "ONNX output embedding {embedding_index} has invalid L2 norm {norm}"
    );
    Ok(embedding
        .iter()
        .map(|component| *component / norm)
        .collect())
}

async fn health(State(text_embedder): State<Arc<OnnxTextEmbedder>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model_id: text_embedder.config.model_id().to_owned(),
        revision: text_embedder.config.model_revision().to_owned(),
        dim: text_embedder.config.embedding_dimension(),
        device: "cpu",
        modalities: ["text"],
        session_count: text_embedder.config.session_count(),
        intra_operation_threads_per_session: text_embedder.config.intra_operation_threads(),
        maximum_pending_requests: text_embedder.config.maximum_pending_requests(),
    })
}

async fn openai_embed(
    State(text_embedder): State<Arc<OnnxTextEmbedder>>,
    request: std::result::Result<Json<OpenAiEmbeddingRequest>, JsonRejection>,
) -> Result<Json<OpenAiEmbeddingResponse>, OpenAiEmbeddingHttpError> {
    let Json(request) = request.map_err(OpenAiEmbeddingHttpError::invalid_json)?;
    let validated_request = request.into_validated_request(
        text_embedder.config.model_id(),
        text_embedder.config.embedding_dimension(),
        text_embedder.config.maximum_batch_size(),
    )?;
    let embedding_batch =
        execute_text_embedding_batch(Arc::clone(&text_embedder), validated_request.text_batch)
            .await
            .map_err(OpenAiEmbeddingHttpError::from)?;

    Ok(Json(OpenAiEmbeddingResponse::from_embedding_batch(
        embedding_batch,
        validated_request.encoding_format,
        text_embedder.config.model_id(),
    )))
}

async fn execute_text_embedding_batch(
    text_embedder: Arc<OnnxTextEmbedder>,
    text_batch: TextEmbeddingBatch,
) -> std::result::Result<TextEmbeddingBatchOutput, TextEmbeddingExecutionError> {
    let admission_permit = match text_embedder.admission_controller.try_admit() {
        InferenceAdmission::Admitted(admission_permit) => admission_permit,
        InferenceAdmission::Saturated => {
            return Err(TextEmbeddingExecutionError::Saturated);
        }
    };
    let text_embedder_for_inference = Arc::clone(&text_embedder);
    tokio::task::spawn_blocking(move || {
        text_embedder_for_inference.embed_text_batch(text_batch, admission_permit)
    })
    .await
    .map_err(|join_error| TextEmbeddingExecutionError::Internal(anyhow::Error::from(join_error)))?
    .map_err(TextEmbeddingExecutionError::from)
}

/// A text batch accepted by the public HTTP boundary and safe for inference.
#[derive(Debug)]
enum TextEmbeddingBatch {
    Texts(Vec<String>),
    TokenIds(Vec<Vec<u32>>),
}

impl TextEmbeddingBatch {
    fn from_texts(
        texts: Vec<String>,
        maximum_batch_size: usize,
    ) -> Result<Self, TextEmbeddingRequestRejection> {
        if texts.is_empty() {
            return Err(TextEmbeddingRequestRejection::EmptyBatch);
        }
        if texts.len() > maximum_batch_size {
            return Err(TextEmbeddingRequestRejection::BatchTooLarge {
                actual_count: texts.len(),
                maximum_count: maximum_batch_size,
            });
        }
        Ok(Self::Texts(texts))
    }

    fn from_token_ids(
        token_sequences: Vec<Vec<u32>>,
        maximum_batch_size: usize,
    ) -> Result<Self, TextEmbeddingRequestRejection> {
        if token_sequences.is_empty() {
            return Err(TextEmbeddingRequestRejection::EmptyBatch);
        }
        if token_sequences.len() > maximum_batch_size {
            return Err(TextEmbeddingRequestRejection::BatchTooLarge {
                actual_count: token_sequences.len(),
                maximum_count: maximum_batch_size,
            });
        }
        if let Some(empty_input_index) = token_sequences.iter().position(Vec::is_empty) {
            return Err(TextEmbeddingRequestRejection::EmptyTokenSequence {
                input_index: empty_input_index,
            });
        }
        Ok(Self::TokenIds(token_sequences))
    }
}

#[derive(Debug, Error)]
enum TextEmbeddingRequestRejection {
    #[error("embedding input must not be empty")]
    EmptyBatch,
    #[error(
        "text batch has {actual_count} inputs, but SIGLIP_MAX_TEXT_BATCH_SIZE is {maximum_count}"
    )]
    BatchTooLarge {
        actual_count: usize,
        maximum_count: usize,
    },
    #[error("token input at index {input_index} must not be empty")]
    EmptyTokenSequence { input_index: usize },
    #[error("token input at index {input_index} contains unknown token id {token_id}")]
    UnknownTokenId { input_index: usize, token_id: u32 },
}

#[derive(Debug, Error)]
enum TextEmbeddingExecutionError {
    #[error("SigLIP 2 text inference capacity is saturated; retry later")]
    Saturated,
    #[error(transparent)]
    Rejected(#[from] TextEmbeddingRequestRejection),
    #[error(transparent)]
    Internal(anyhow::Error),
}

impl From<TextEmbeddingOperationError> for TextEmbeddingExecutionError {
    fn from(error: TextEmbeddingOperationError) -> Self {
        match error {
            TextEmbeddingOperationError::Rejected(rejection) => Self::Rejected(rejection),
            TextEmbeddingOperationError::Inference(error) => Self::Internal(error),
        }
    }
}

#[derive(Deserialize)]
struct OpenAiEmbeddingRequest {
    input: OpenAiEmbeddingInput,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    encoding_format: OpenAiEmbeddingEncodingFormat,
    #[serde(default)]
    dimensions: Option<usize>,
}

impl OpenAiEmbeddingRequest {
    fn into_validated_request(
        self,
        served_model_id: &str,
        embedding_dimension: usize,
        maximum_batch_size: usize,
    ) -> std::result::Result<ValidatedOpenAiEmbeddingRequest, OpenAiEmbeddingRequestRejection> {
        if let Some(requested_model_id) = self.model
            && requested_model_id != served_model_id
        {
            return Err(OpenAiEmbeddingRequestRejection::ModelMismatch {
                requested_model_id,
                served_model_id: served_model_id.to_owned(),
            });
        }
        if let Some(requested_dimension) = self.dimensions
            && requested_dimension != embedding_dimension
        {
            return Err(OpenAiEmbeddingRequestRejection::UnsupportedDimension {
                requested_dimension,
                embedding_dimension,
            });
        }

        let text_batch = match self.input {
            OpenAiEmbeddingInput::SingleText(text) => {
                TextEmbeddingBatch::from_texts(vec![text], maximum_batch_size)?
            }
            OpenAiEmbeddingInput::TextBatch(texts) => {
                TextEmbeddingBatch::from_texts(texts, maximum_batch_size)?
            }
            OpenAiEmbeddingInput::SingleTokenSequence(token_ids) => {
                TextEmbeddingBatch::from_token_ids(vec![token_ids], maximum_batch_size)?
            }
            OpenAiEmbeddingInput::TokenSequenceBatch(token_sequences) => {
                TextEmbeddingBatch::from_token_ids(token_sequences, maximum_batch_size)?
            }
        };

        Ok(ValidatedOpenAiEmbeddingRequest {
            text_batch,
            encoding_format: self.encoding_format,
        })
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiEmbeddingInput {
    SingleText(String),
    TextBatch(Vec<String>),
    SingleTokenSequence(Vec<u32>),
    TokenSequenceBatch(Vec<Vec<u32>>),
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAiEmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug)]
struct ValidatedOpenAiEmbeddingRequest {
    text_batch: TextEmbeddingBatch,
    encoding_format: OpenAiEmbeddingEncodingFormat,
}

#[derive(Debug, Error)]
enum OpenAiEmbeddingRequestRejection {
    #[error(transparent)]
    InvalidInput(#[from] TextEmbeddingRequestRejection),
    #[error(
        "requested model {requested_model_id:?} does not match served model {served_model_id:?}"
    )]
    ModelMismatch {
        requested_model_id: String,
        served_model_id: String,
    },
    #[error(
        "requested embedding dimension {requested_dimension} is unsupported; this model returns {embedding_dimension} dimensions"
    )]
    UnsupportedDimension {
        requested_dimension: usize,
        embedding_dimension: usize,
    },
}

#[derive(Serialize)]
struct OpenAiEmbeddingResponse {
    object: &'static str,
    data: Vec<OpenAiEmbeddingData>,
    model: String,
    usage: OpenAiEmbeddingUsage,
}

impl OpenAiEmbeddingResponse {
    fn from_embedding_batch(
        embedding_batch: TextEmbeddingBatchOutput,
        encoding_format: OpenAiEmbeddingEncodingFormat,
        model_id: &str,
    ) -> Self {
        let data = embedding_batch
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| OpenAiEmbeddingData {
                object: "embedding",
                embedding: encode_openai_embedding(embedding, encoding_format),
                index,
            })
            .collect();
        Self {
            object: "list",
            data,
            model: model_id.to_owned(),
            usage: OpenAiEmbeddingUsage {
                prompt_tokens: embedding_batch.input_token_count,
                total_tokens: embedding_batch.input_token_count,
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAiEmbeddingData {
    object: &'static str,
    embedding: OpenAiEmbeddingValues,
    index: usize,
}

#[derive(Serialize)]
#[serde(untagged)]
enum OpenAiEmbeddingValues {
    Float(Vec<f32>),
    Base64(String),
}

fn encode_openai_embedding(
    embedding: Vec<f32>,
    encoding_format: OpenAiEmbeddingEncodingFormat,
) -> OpenAiEmbeddingValues {
    match encoding_format {
        OpenAiEmbeddingEncodingFormat::Float => OpenAiEmbeddingValues::Float(embedding),
        OpenAiEmbeddingEncodingFormat::Base64 => {
            let mut embedding_bytes = Vec::with_capacity(embedding.len() * size_of::<f32>());
            for component in embedding {
                embedding_bytes.extend_from_slice(&component.to_le_bytes());
            }
            OpenAiEmbeddingValues::Base64(BASE64_STANDARD.encode(embedding_bytes))
        }
    }
}

#[derive(Serialize)]
struct OpenAiEmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug)]
struct OpenAiEmbeddingHttpError(OpenAiEmbeddingHttpErrorKind);

#[derive(Debug)]
enum OpenAiEmbeddingHttpErrorKind {
    Empty(String),
    Validation(String),
    BatchTooLarge(String),
    Overloaded,
    Backend(anyhow::Error),
}

impl OpenAiEmbeddingHttpError {
    fn invalid_json(rejection: JsonRejection) -> Self {
        Self(OpenAiEmbeddingHttpErrorKind::Validation(
            rejection.body_text(),
        ))
    }
}

impl From<OpenAiEmbeddingRequestRejection> for OpenAiEmbeddingHttpError {
    fn from(rejection: OpenAiEmbeddingRequestRejection) -> Self {
        match rejection {
            OpenAiEmbeddingRequestRejection::InvalidInput(
                TextEmbeddingRequestRejection::EmptyBatch,
            ) => Self(OpenAiEmbeddingHttpErrorKind::Empty(rejection.to_string())),
            OpenAiEmbeddingRequestRejection::InvalidInput(
                TextEmbeddingRequestRejection::BatchTooLarge { .. },
            ) => Self(OpenAiEmbeddingHttpErrorKind::BatchTooLarge(
                rejection.to_string(),
            )),
            rejection => Self(OpenAiEmbeddingHttpErrorKind::Validation(
                rejection.to_string(),
            )),
        }
    }
}

impl From<TextEmbeddingExecutionError> for OpenAiEmbeddingHttpError {
    fn from(error: TextEmbeddingExecutionError) -> Self {
        match error {
            TextEmbeddingExecutionError::Saturated => {
                Self(OpenAiEmbeddingHttpErrorKind::Overloaded)
            }
            TextEmbeddingExecutionError::Rejected(rejection) => Self(
                OpenAiEmbeddingHttpErrorKind::Validation(rejection.to_string()),
            ),
            TextEmbeddingExecutionError::Internal(error) => {
                Self(OpenAiEmbeddingHttpErrorKind::Backend(error))
            }
        }
    }
}

impl IntoResponse for OpenAiEmbeddingHttpError {
    fn into_response(self) -> Response {
        let (status_code, message, error_type, retry_after) = match self.0 {
            OpenAiEmbeddingHttpErrorKind::Empty(message) => {
                (StatusCode::BAD_REQUEST, message, "empty", None)
            }
            OpenAiEmbeddingHttpErrorKind::Validation(message) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                message,
                "validation",
                None,
            ),
            OpenAiEmbeddingHttpErrorKind::BatchTooLarge(message) => {
                (StatusCode::PAYLOAD_TOO_LARGE, message, "validation", None)
            }
            OpenAiEmbeddingHttpErrorKind::Overloaded => (
                StatusCode::TOO_MANY_REQUESTS,
                "SigLIP 2 text inference capacity is saturated; retry later".to_owned(),
                "overloaded",
                Some("1"),
            ),
            OpenAiEmbeddingHttpErrorKind::Backend(error) => {
                tracing::error!(error = %error, "SigLIP 2 ONNX inference failed");
                (
                    StatusCode::FAILED_DEPENDENCY,
                    "SigLIP 2 text inference failed".to_owned(),
                    "backend",
                    None,
                )
            }
        };
        let mut response = (
            status_code,
            Json(OpenAiEmbeddingErrorResponse {
                message,
                code: status_code.as_u16(),
                error_type,
            }),
        )
            .into_response();
        if let Some(retry_after) = retry_after {
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from_static(retry_after));
        }
        response
    }
}

#[derive(Serialize)]
struct OpenAiEmbeddingErrorResponse {
    message: String,
    code: u16,
    #[serde(rename = "type")]
    error_type: &'static str,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model_id: String,
    revision: String,
    dim: usize,
    device: &'static str,
    modalities: [&'static str; 1],
    session_count: usize,
    intra_operation_threads_per_session: usize,
    maximum_pending_requests: usize,
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use axum::http::{StatusCode, header::RETRY_AFTER};
    use axum::response::IntoResponse;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use serde_json::json;

    use super::{
        InferenceAdmission, InferenceAdmissionController, InferenceAdmissionPermit,
        OpenAiEmbeddingEncodingFormat, OpenAiEmbeddingHttpError, OpenAiEmbeddingInput,
        OpenAiEmbeddingRequest, OpenAiEmbeddingRequestRejection, OpenAiEmbeddingResponse,
        OpenAiEmbeddingValues, TextEmbeddingBatch, TextEmbeddingBatchOutput,
        TextEmbeddingExecutionError, TextEmbeddingRequestRejection, normalize_embedding_batch,
    };

    #[test]
    fn inference_admission_bounds_active_and_pending_requests() {
        let admission_controller =
            InferenceAdmissionController::new(2, 1).expect("capacity must be representable");
        let first_permit = expect_admitted(admission_controller.try_admit());
        let _second_permit = expect_admitted(admission_controller.try_admit());
        let _pending_permit = expect_admitted(admission_controller.try_admit());

        assert!(matches!(
            admission_controller.try_admit(),
            InferenceAdmission::Saturated
        ));

        drop(first_permit);
        let _replacement_permit = expect_admitted(admission_controller.try_admit());
    }

    #[test]
    fn openai_saturation_uses_the_tei_overloaded_response() {
        let response =
            OpenAiEmbeddingHttpError::from(TextEmbeddingExecutionError::Saturated).into_response();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "1");
    }

    #[test]
    fn empty_text_batch_is_rejected_at_the_http_boundary() {
        let rejection = TextEmbeddingBatch::from_texts(Vec::new(), 32)
            .expect_err("empty text batches cannot be inferred");

        assert!(matches!(
            rejection,
            TextEmbeddingRequestRejection::EmptyBatch
        ));
    }

    #[test]
    fn oversized_text_batch_is_rejected_at_the_http_boundary() {
        let rejection = TextEmbeddingBatch::from_texts(vec!["one".to_owned(), "two".to_owned()], 1)
            .expect_err("batch limit must protect the CPU runtime");

        assert!(matches!(
            rejection,
            TextEmbeddingRequestRejection::BatchTooLarge {
                actual_count: 2,
                maximum_count: 1,
            }
        ));
    }

    #[test]
    fn openai_text_batch_is_parsed_without_changing_input_order() {
        let request: OpenAiEmbeddingRequest = serde_json::from_value(json!({
            "model": "google/siglip2-so400m-patch16-384",
            "input": ["first query", "second query"]
        }))
        .expect("OpenAI text batch must deserialize");

        let validated_request = request
            .into_validated_request("google/siglip2-so400m-patch16-384", 1152, 32)
            .expect("matching model and dimension must be accepted");

        match validated_request.text_batch {
            TextEmbeddingBatch::Texts(texts) => {
                assert_eq!(texts, vec!["first query", "second query"]);
            }
            TextEmbeddingBatch::TokenIds(_) => panic!("text input must remain text"),
        }
    }

    #[test]
    fn openai_token_id_batches_are_a_distinct_valid_input_variant() {
        let request: OpenAiEmbeddingRequest = serde_json::from_value(json!({
            "input": [[1, 2, 3], [4, 5]],
            "encoding_format": "base64"
        }))
        .expect("OpenAI token ID batch must deserialize");

        let validated_request = request
            .into_validated_request("google/siglip2-so400m-patch16-384", 1152, 32)
            .expect("non-empty token ID batches must be accepted at the HTTP boundary");

        assert!(matches!(
            validated_request.text_batch,
            TextEmbeddingBatch::TokenIds(token_sequences)
                if token_sequences == vec![vec![1, 2, 3], vec![4, 5]]
        ));
        assert!(matches!(
            validated_request.encoding_format,
            OpenAiEmbeddingEncodingFormat::Base64
        ));
    }

    #[test]
    fn openai_request_rejects_a_different_embedding_model() {
        let request = OpenAiEmbeddingRequest {
            input: OpenAiEmbeddingInput::SingleText("query".to_owned()),
            model: Some("different/model".to_owned()),
            encoding_format: OpenAiEmbeddingEncodingFormat::Float,
            dimensions: None,
        };

        let rejection = request
            .into_validated_request("google/siglip2-so400m-patch16-384", 1152, 32)
            .expect_err("cross-model requests must fail closed");

        assert!(matches!(
            rejection,
            OpenAiEmbeddingRequestRejection::ModelMismatch { .. }
        ));
    }

    #[test]
    fn openai_request_rejects_dimension_changes() {
        let request = OpenAiEmbeddingRequest {
            input: OpenAiEmbeddingInput::SingleText("query".to_owned()),
            model: None,
            encoding_format: OpenAiEmbeddingEncodingFormat::Float,
            dimensions: Some(256),
        };

        let rejection = request
            .into_validated_request("google/siglip2-so400m-patch16-384", 1152, 32)
            .expect_err("SigLIP 2 does not support output-dimension projection");

        assert!(matches!(
            rejection,
            OpenAiEmbeddingRequestRejection::UnsupportedDimension {
                requested_dimension: 256,
                embedding_dimension: 1152,
            }
        ));
    }

    #[test]
    fn openai_base64_response_preserves_little_endian_f32_values_and_usage() {
        let response = OpenAiEmbeddingResponse::from_embedding_batch(
            TextEmbeddingBatchOutput {
                embeddings: vec![vec![0.0, -0.0, 1.25]],
                input_token_count: 7,
            },
            OpenAiEmbeddingEncodingFormat::Base64,
            "google/siglip2-so400m-patch16-384",
        );

        assert_eq!(response.object, "list");
        assert_eq!(response.model, "google/siglip2-so400m-patch16-384");
        assert_eq!(response.usage.prompt_tokens, 7);
        assert_eq!(response.usage.total_tokens, 7);
        assert_eq!(response.data[0].index, 0);
        let OpenAiEmbeddingValues::Base64(encoded_embedding) = &response.data[0].embedding else {
            panic!("base64 encoding must return a base64 string");
        };
        let embedding_bytes = BASE64_STANDARD
            .decode(encoded_embedding)
            .expect("response must contain valid base64");
        let returned_values: Vec<f32> = embedding_bytes
            .chunks_exact(size_of::<f32>())
            .map(|component_bytes| {
                f32::from_le_bytes(
                    component_bytes
                        .try_into()
                        .expect("each encoded component is exactly four bytes"),
                )
            })
            .collect();

        assert_eq!(
            returned_values
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [0.0_f32, -0.0, 1.25]
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalized_embeddings_have_unit_norm() {
        let normalized_embeddings =
            normalize_embedding_batch(&[3.0, 4.0], 1, 2).expect("finite nonzero vector is valid");

        assert_eq!(normalized_embeddings, vec![vec![0.6, 0.8]]);
    }

    #[test]
    fn zero_vector_is_rejected() {
        let error = normalize_embedding_batch(&[0.0, 0.0], 1, 2)
            .expect_err("zero embedding cannot satisfy the service contract");

        assert!(error.to_string().contains("invalid L2 norm"));
    }

    #[test]
    fn output_dimension_mismatch_is_rejected() {
        let error = normalize_embedding_batch(&[1.0, 2.0], 1, 3)
            .expect_err("unexpected ONNX output shape must fail closed");

        assert!(
            error
                .to_string()
                .contains("expected 1 vectors of dimension 3")
        );
    }

    fn expect_admitted(admission: InferenceAdmission) -> InferenceAdmissionPermit {
        match admission {
            InferenceAdmission::Admitted(admission_permit) => admission_permit,
            InferenceAdmission::Saturated => panic!("expected inference capacity to be available"),
        }
    }
}
