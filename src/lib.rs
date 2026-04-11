pub mod core;
// pub mod effects;
pub mod engine;
pub mod geometry;
pub mod input;
pub mod rendering;
pub mod scene;
pub mod shaders;
#[cfg(test)]
pub mod tests;

pub use core::collisions::CollisionType;
pub use core::{Material, MaterialBuilder, Physics, ShaderType, Transform};
pub use engine::{Engine, PerspectiveCamera};
pub use geometry::Mesh;
pub use rendering::compute_registry::ComputeShaderType;
