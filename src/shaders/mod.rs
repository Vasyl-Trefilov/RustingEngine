//! Shader module - Contains all GPU shaders for the rendering engine.
//!
//! This module exports shader modules that are compiled at build time:
//! - Vertex shaders: Transform vertices and prepare for rasterization
//! - Fragment shaders: Calculate pixel colors with PBR lighting
//! - Compute shaders: Physics simulation and culling
//!
//! Shaders are written in GLSL and loaded via vulkano-shaders.
//! Each shader type has multiple variants for different rendering needs.

pub mod compute;
pub mod fragment;
pub mod vertex;
