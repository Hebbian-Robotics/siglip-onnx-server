//! CPU-only, text-only `SigLIP 2` ONNX embedding service.
//!
//! This crate intentionally exposes no image embedding route. It is suitable
//! only for live-query embedding against indexes whose image vectors were
//! produced by the GPU `SigLIP 2` service.

pub mod config;
pub mod server;
