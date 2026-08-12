//! Environment configuration and model-contract validation.

use std::env;
use std::fs;
use std::num::{NonZeroU16, NonZeroUsize};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// Process configuration resolved and validated once during startup.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    port: ServicePort,
    model_path: ExistingFilePath,
    tokenizer_path: ExistingFilePath,
    onnx_runtime_library_path: ExistingFilePath,
    output_name: OnnxOutputName,
    maximum_batch_size: PositiveCount,
    session_count: PositiveCount,
    intra_operation_threads: PositiveCount,
    maximum_pending_requests: NonNegativeCount,
    model_contract: ModelContract,
}

impl ServiceConfig {
    /// Reads the service configuration. Every model/runtime path is required
    /// so startup cannot accidentally fetch mutable artifacts from the network.
    ///
    /// # Errors
    ///
    /// Returns an error when a required path is missing, a configured numeric
    /// value is invalid, an artifact file does not exist, or the model manifest
    /// does not describe a complete immutable embedding contract.
    pub fn from_environment() -> Result<Self> {
        let model_manifest_path = ExistingFilePath::from_environment("SIGLIP_MODEL_MANIFEST_PATH")?;
        let model_contract = ModelContract::load(model_manifest_path.as_path())?;

        Ok(Self {
            port: ServicePort::from_environment("SIGLIP_PORT", 8000)?,
            model_path: ExistingFilePath::from_environment("SIGLIP_ONNX_MODEL_PATH")?,
            tokenizer_path: ExistingFilePath::from_environment("SIGLIP_TOKENIZER_PATH")?,
            onnx_runtime_library_path: ExistingFilePath::from_environment(
                "SIGLIP_ORT_LIBRARY_PATH",
            )?,
            output_name: OnnxOutputName::from_environment(
                "SIGLIP_ONNX_OUTPUT_NAME",
                "text_embeds",
            )?,
            maximum_batch_size: PositiveCount::from_environment("SIGLIP_MAX_TEXT_BATCH_SIZE", 32)?,
            session_count: PositiveCount::from_environment("SIGLIP_ORT_SESSION_COUNT", 1)?,
            intra_operation_threads: PositiveCount::from_environment(
                "SIGLIP_ORT_INTRA_OP_THREADS",
                1,
            )?,
            maximum_pending_requests: NonNegativeCount::from_environment(
                "SIGLIP_MAX_PENDING_TEXT_REQUESTS",
                32,
            )?,
            model_contract,
        })
    }

    pub fn port(&self) -> u16 {
        self.port.get()
    }

    pub fn model_path(&self) -> &Path {
        self.model_path.as_path()
    }

    pub fn tokenizer_path(&self) -> &Path {
        self.tokenizer_path.as_path()
    }

    pub fn onnx_runtime_library_path(&self) -> &Path {
        self.onnx_runtime_library_path.as_path()
    }

    pub fn output_name(&self) -> &str {
        self.output_name.as_str()
    }

    pub fn maximum_batch_size(&self) -> usize {
        self.maximum_batch_size.get()
    }

    pub fn session_count(&self) -> usize {
        self.session_count.get()
    }

    pub fn intra_operation_threads(&self) -> usize {
        self.intra_operation_threads.get()
    }

    pub fn maximum_pending_requests(&self) -> usize {
        self.maximum_pending_requests.get()
    }

    pub fn model_id(&self) -> &str {
        self.model_contract.model_id()
    }

    pub fn model_revision(&self) -> &str {
        self.model_contract.model_revision()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.model_contract.embedding_dimension()
    }

    pub fn text_max_token_length(&self) -> usize {
        self.model_contract.text_max_token_length()
    }

    pub fn pad_token(&self) -> &str {
        self.model_contract.pad_token()
    }
}

/// A model identity and vector shape that cannot be partially configured.
#[derive(Clone, Debug)]
struct ModelContract {
    model_id: ModelIdentifier,
    model_revision: ModelRevision,
    embedding_dimension: PositiveCount,
    text_max_token_length: PositiveCount,
    pad_token: PadToken,
}

#[derive(Debug, Deserialize)]
struct ModelManifestDocument {
    repository: String,
    revision: String,
    embedding_dimension: usize,
    text_max_token_length: usize,
    #[serde(default)]
    pad_token: Option<String>,
}

impl ModelContract {
    fn load(manifest_path: &Path) -> Result<Self> {
        let manifest_json = fs::read_to_string(manifest_path)
            .with_context(|| format!("reading model manifest {}", manifest_path.display()))?;
        Self::from_json(&manifest_json)
            .with_context(|| format!("validating model manifest {}", manifest_path.display()))
    }

    fn from_json(manifest_json: &str) -> Result<Self> {
        let document: ModelManifestDocument =
            serde_json::from_str(manifest_json).context("parsing model manifest JSON")?;
        Ok(Self {
            model_id: ModelIdentifier::parse(document.repository)?,
            model_revision: ModelRevision::parse(document.revision)?,
            embedding_dimension: PositiveCount::parse(
                "model manifest embedding_dimension",
                &document.embedding_dimension.to_string(),
            )?,
            text_max_token_length: PositiveCount::parse(
                "model manifest text_max_token_length",
                &document.text_max_token_length.to_string(),
            )?,
            pad_token: PadToken::parse(document.pad_token)?,
        })
    }

    fn model_id(&self) -> &str {
        self.model_id.as_str()
    }

    fn model_revision(&self) -> &str {
        self.model_revision.as_str()
    }

    fn embedding_dimension(&self) -> usize {
        self.embedding_dimension.get()
    }

    fn text_max_token_length(&self) -> usize {
        self.text_max_token_length.get()
    }

    fn pad_token(&self) -> &str {
        self.pad_token.as_str()
    }
}

/// The padding token literal the paired tokenizer defines. Tokenizer families
/// spell this differently, so it is part of the pinned model contract for the
/// same reason the vector dimension and token length are: a value the
/// artifacts decide does not belong in the code.
///
/// The field is optional and defaults to `SigLIP 2`'s `<pad>` so manifests
/// written before it existed keep working. That default is safe rather than
/// merely convenient, because a tokenizer that does not define `<pad>` fails
/// when it is loaded; an unstated pad token cannot silently pad with the wrong
/// one. Make it required once every deployed manifest declares it.
#[derive(Clone, Debug)]
struct PadToken(String);

impl PadToken {
    const SIGLIP_PAD_TOKEN: &'static str = "<pad>";

    fn parse(raw_value: Option<String>) -> Result<Self> {
        let Some(raw_value) = raw_value else {
            return Ok(Self(Self::SIGLIP_PAD_TOKEN.to_owned()));
        };
        let pad_token = raw_value.trim().to_owned();
        ensure!(
            !pad_token.is_empty(),
            "model manifest pad_token must be non-empty when present"
        );
        Ok(Self(pad_token))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
struct ModelIdentifier(String);

impl ModelIdentifier {
    fn parse(raw_value: String) -> Result<Self> {
        let model_id = raw_value.trim().to_owned();
        ensure!(
            !model_id.is_empty(),
            "model manifest repository must be non-empty"
        );
        Ok(Self(model_id))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
struct ModelRevision(String);

impl ModelRevision {
    fn parse(raw_value: String) -> Result<Self> {
        let model_revision = raw_value.trim().to_owned();
        ensure!(
            model_revision.len() == 40
                && model_revision
                    .bytes()
                    .all(|character| character.is_ascii_hexdigit()),
            "model manifest revision must be a 40-character hexadecimal commit SHA"
        );
        Ok(Self(model_revision))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// A TCP port that has already been checked to be usable by a listener.
#[derive(Clone, Copy, Debug)]
struct ServicePort(NonZeroU16);

impl ServicePort {
    fn from_environment(variable_name: &str, default_value: u16) -> Result<Self> {
        match read_nonempty(variable_name) {
            Some(raw_value) => Self::parse(variable_name, &raw_value),
            None => Self::parse(variable_name, &default_value.to_string()),
        }
    }

    fn parse(variable_name: &str, raw_value: &str) -> Result<Self> {
        let parsed_value = raw_value
            .parse::<u16>()
            .with_context(|| format!("{variable_name} must be a TCP port"))?;
        let nonzero_value = NonZeroU16::new(parsed_value)
            .with_context(|| format!("{variable_name} must be greater than zero"))?;
        Ok(Self(nonzero_value))
    }

    fn get(self) -> u16 {
        self.0.get()
    }
}

/// A nonzero operational limit, used for model and CPU resource budgets.
#[derive(Clone, Copy, Debug)]
struct PositiveCount(NonZeroUsize);

impl PositiveCount {
    fn from_environment(variable_name: &str, default_value: usize) -> Result<Self> {
        match read_nonempty(variable_name) {
            Some(raw_value) => Self::parse(variable_name, &raw_value),
            None => Self::parse(variable_name, &default_value.to_string()),
        }
    }

    fn parse(variable_name: &str, raw_value: &str) -> Result<Self> {
        let parsed_value = raw_value
            .parse::<usize>()
            .with_context(|| format!("{variable_name} must be a positive integer"))?;
        let nonzero_value = NonZeroUsize::new(parsed_value)
            .with_context(|| format!("{variable_name} must be greater than zero"))?;
        Ok(Self(nonzero_value))
    }

    fn get(self) -> usize {
        self.0.get()
    }
}

/// An operational limit where zero deliberately disables pending work.
#[derive(Clone, Copy, Debug)]
struct NonNegativeCount(usize);

impl NonNegativeCount {
    fn from_environment(variable_name: &str, default_value: usize) -> Result<Self> {
        match read_nonempty(variable_name) {
            Some(raw_value) => Self::parse(variable_name, &raw_value),
            None => Ok(Self(default_value)),
        }
    }

    fn parse(variable_name: &str, raw_value: &str) -> Result<Self> {
        let parsed_value = raw_value
            .parse::<usize>()
            .with_context(|| format!("{variable_name} must be a non-negative integer"))?;
        Ok(Self(parsed_value))
    }

    fn get(self) -> usize {
        self.0
    }
}

/// A local artifact that has been checked to be a regular file at startup.
#[derive(Clone, Debug)]
struct ExistingFilePath(PathBuf);

impl ExistingFilePath {
    fn from_environment(variable_name: &str) -> Result<Self> {
        let raw_value =
            read_nonempty(variable_name).with_context(|| format!("{variable_name} is required"))?;
        let path = PathBuf::from(raw_value);
        ensure!(
            path.is_file(),
            "{variable_name} must name a readable file, got {}",
            path.display()
        );
        Ok(Self(path))
    }

    fn as_path(&self) -> &Path {
        &self.0
    }
}

/// The named tensor returned by the configured text-tower ONNX graph.
#[derive(Clone, Debug)]
struct OnnxOutputName(String);

impl OnnxOutputName {
    fn from_environment(variable_name: &str, default_value: &str) -> Result<Self> {
        let raw_value = read_nonempty(variable_name).unwrap_or_else(|| default_value.to_owned());
        Self::parse(variable_name, raw_value)
    }

    fn parse(variable_name: &str, value: String) -> Result<Self> {
        ensure!(
            !value.is_empty(),
            "{variable_name} must be a non-empty ONNX output name"
        );
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

fn read_nonempty(variable_name: &str) -> Option<String> {
    env::var(variable_name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{ModelContract, NonNegativeCount, PositiveCount, ServicePort};

    const VALID_MODEL_MANIFEST: &str = r#"{
        "repository": "organization/siglip-model",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "index_model_id": "siglip/organization/siglip-model",
        "embedding_dimension": 1152,
        "text_max_token_length": 64,
        "pad_token": "<pad>"
    }"#;

    #[test]
    fn model_manifest_produces_a_complete_validated_contract() {
        let model_contract =
            ModelContract::from_json(VALID_MODEL_MANIFEST).expect("valid manifest must parse");

        assert_eq!(model_contract.model_id(), "organization/siglip-model");
        assert_eq!(
            model_contract.model_revision(),
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(model_contract.embedding_dimension(), 1152);
        assert_eq!(model_contract.text_max_token_length(), 64);
        assert_eq!(model_contract.pad_token(), "<pad>");
    }

    #[test]
    fn declared_pad_token_overrides_the_default() {
        let manifest_with_other_pad_token = VALID_MODEL_MANIFEST.replace("\"<pad>\"", "\"[PAD]\"");

        let model_contract = ModelContract::from_json(&manifest_with_other_pad_token)
            .expect("a checkpoint may spell its pad token differently");

        assert_eq!(model_contract.pad_token(), "[PAD]");
    }

    /// Manifests written before `pad_token` existed must keep working. The
    /// default cannot mask a mismatch, because a tokenizer that does not
    /// define `<pad>` fails when it is loaded.
    #[test]
    fn absent_pad_token_defaults_to_the_siglip_literal() {
        let manifest_without_pad_token =
            VALID_MODEL_MANIFEST.replace(",\n        \"pad_token\": \"<pad>\"", "");

        let model_contract = ModelContract::from_json(&manifest_without_pad_token)
            .expect("manifests predating the pad_token field must still load");

        assert_eq!(model_contract.pad_token(), "<pad>");
    }

    #[test]
    fn empty_pad_token_is_rejected() {
        let manifest_with_empty_pad_token = VALID_MODEL_MANIFEST.replace("\"<pad>\"", "\"  \"");

        let error = ModelContract::from_json(&manifest_with_empty_pad_token)
            .expect_err("a blank pad token cannot name a real tokenizer entry");

        assert!(error.to_string().contains("must be non-empty"));
    }

    #[test]
    fn mutable_model_revision_is_rejected() {
        let mutable_revision_manifest =
            VALID_MODEL_MANIFEST.replace("0123456789abcdef0123456789abcdef01234567", "main");

        let error = ModelContract::from_json(&mutable_revision_manifest)
            .expect_err("a mutable model revision must fail closed");

        assert!(error.to_string().contains("commit SHA"));
    }

    #[test]
    fn zero_embedding_dimension_is_rejected() {
        let zero_dimension_manifest = VALID_MODEL_MANIFEST.replace(
            "\"embedding_dimension\": 1152",
            "\"embedding_dimension\": 0",
        );

        let error = ModelContract::from_json(&zero_dimension_manifest)
            .expect_err("zero-dimensional embeddings cannot be served");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn zero_port_is_rejected() {
        let error = ServicePort::parse("SIGLIP_PORT", "0").expect_err("port zero must be invalid");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn zero_thread_count_is_rejected() {
        let error = PositiveCount::parse("SIGLIP_ORT_INTRA_OP_THREADS", "0")
            .expect_err("zero threads must be invalid");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn zero_pending_requests_disables_the_waiting_queue() {
        let maximum_pending_requests =
            NonNegativeCount::parse("SIGLIP_MAX_PENDING_TEXT_REQUESTS", "0")
                .expect("zero is a valid pending-request limit");

        assert_eq!(maximum_pending_requests.get(), 0);
    }
}
