use crate::shapes::Mesh;
use crate::scene::animation::AnimationType;
use nalgebra::{Matrix4, Rotation3, Vector3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4], // 64 bytes
    pub color: [f32; 3], // 12 bytes
    pub padding: f32,   // hahaha, so basically, this shit is useless, but we need it, bc I said(we really need it to keep data in 16 bytes for GPU)
    pub shininess: f32, // 4 bytes (Total 80)
    pub specular_strength: f32, // 4 bytes
    pub roughness: f32, // 4 bytes
    pub metalness: f32, // 4 bytes
    // Total 96, perfect for GPU, bc 96 mod 16 is 0
}

#[derive(Clone)]
pub struct Instance {
    pub transform: Transform,
    pub color: [f32; 3],
    pub shininess: f32,          
    pub specular_strength: f32,  
    pub roughness: f32,    
    pub metalness: f32,
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
            specular_strength: 0.5,
            shininess: 32.0,
            roughness: 0.5,
            metalness: 0.5
        }
    }
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