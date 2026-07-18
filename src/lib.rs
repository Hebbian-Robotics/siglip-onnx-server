//! CPU-only, text-only `SigLIP` ONNX embedding service.
//!
//! This crate intentionally exposes no image embedding route. It is suitable
//! only for live-query embedding against indexes whose image vectors were
//! produced by the GPU `SigLIP` service.

pub mod config;
pub mod server;
