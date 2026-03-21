use crate::shapes::Mesh;
use crate::scene::animation::AnimationType;
use nalgebra::{Matrix4, Rotation3, Vector3};
use vulkano::image::{ImageAccess, view::ImageView};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4], // 64 bytes
    pub color: [f32; 3], // 12 bytes
    pub padding: f32,   // hahaha, so basically, this shit is useless, but we need it, bc I said(we really need it to keep data in 16 bytes for GPU)
    pub mat_props: [f32; 4], // [roughness, metalness, reserved, reserved]
    // Total 96, perfect for GPU, bc 96 mod 16 is 0
}

#[derive(Clone)]
pub struct Instance {
    pub transform: Transform,
    pub color: [f32; 3],
    pub roughness: f32,    
    pub metalness: f32,
    pub base_color_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub animation: AnimationType,
    pub model_matrix: [[f32; 4]; 4],
    pub original_position: [f32; 3],
    pub velocity: [f32; 3],
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            transform: Transform::default(),
            original_position: [0.0, 0.0, 0.0],
            animation: AnimationType::Static,
            velocity: [0.0, 0.0, 0.0],
            model_matrix: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
            color: [1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic_roughness_texture: None,
            roughness: 0.5,
            metalness: 0.5
        }
    }
}

pub struct Texture {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

// ! TRANSFORM - Position, rotation, and scale of an object
#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub position: [f32; 3],  // Translation in world space
    pub rotation: [f32; 3],   // Euler angles, but I like Quaternions, but its complex as hell to implement it everywhere, but I sure that I will do this soon(maybe later then soon)
    pub scale: [f32; 3],      // Scale factors
}

// * It can be used to create a new objects or set default settings, like in blender 'reset transform' or something like that
impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        let translation = Matrix4::new_translation(&Vector3::from(self.position));
        
        let rotation = Rotation3::from_euler_angles(
            self.rotation[0], 
            self.rotation[1], 
            self.rotation[2]
        ).to_homogeneous();
        
        let scale = Matrix4::new_nonuniform_scaling(&Vector3::from(self.scale));
        
        let model_matrix = translation * rotation * scale;
        
        model_matrix.into()
    }
    pub fn from_matrix(m: Matrix4<f32>) -> Self {
        let position = [m[(0, 3)], m[(1, 3)], m[(2, 3)]];

        let scale = [
            Vector3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]).norm(),
            Vector3::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]).norm(),
            Vector3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]).norm(),
        ];

        let rotation_matrix = nalgebra::Matrix3::new(
            m[(0, 0)] / scale[0], m[(0, 1)] / scale[1], m[(0, 2)] / scale[2],
            m[(1, 0)] / scale[0], m[(1, 1)] / scale[1], m[(1, 2)] / scale[2],
            m[(2, 0)] / scale[0], m[(2, 1)] / scale[1], m[(2, 2)] / scale[2],
        );

        let rotation = Rotation3::from_matrix_unchecked(rotation_matrix)
            .euler_angles();

        Self {
            position,
            rotation: [rotation.0, rotation.1, rotation.2],
            scale,
        }
    }
    // * Helper methods for common transformations, but I use dirrect change, bc I did like that all time in THREE.js, but this functions can be pretty useful for someone
    // translate
    pub fn translate(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }

    // rotate
    pub fn rotate(mut self, x: f32, y: f32, z: f32) -> Self {
        self.rotation = [x, y, z];
        self
    }

    // scale
    pub fn scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.scale = [x, y, z];
        self
    }

    // uniform_scale
    pub fn uniform_scale(mut self, s: f32) -> Self {
        self.scale = [s, s, s];
        self
    }

    // I am going crazy
}

pub struct RenderBatch {
    pub mesh: Mesh,
    pub instances: Vec<Instance>,
}