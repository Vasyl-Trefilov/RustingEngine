//shapes.rs
// ! SHAPE AND MESH SYSTEM - Defines geometry and scene organization

use nalgebra::{Matrix4, Rotation3, Vector3};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::command_buffer::AutoCommandBufferBuilder;

// * Trait combining all requirements for vertex types
pub trait VertexType: Vertex + bytemuck::Pod + bytemuck::Zeroable + Send + Sync {}

// ! VERTEX WITH POSITION AND COLOR - Basic vertex format
#[repr(C)]
#[derive(Copy, Clone, Debug, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexPosColor {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],  // XYZ position
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],      // RGB color
    #[format(R32G32B32_SFLOAT)] 
    pub barycentric: [f32; 3], 
}

impl VertexType for VertexPosColor {}

// ! VERTEX WITH UV COORDINATES - For textured objects (future use)
#[repr(C)]
#[derive(Copy, Clone, Debug, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexPosColorUv {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],   // XYZ position
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],       // RGB color
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],          // Texture coordinates

}

impl VertexType for VertexPosColorUv {}

// ! MESH - Container for vertex and index data on the GPU
#[derive(Clone)]
pub struct Mesh {
    pub vertices: Subbuffer<[VertexPosColor]>,   // GPU buffer of vertices
    pub indices: Option<Subbuffer<[u32]>>,       // Optional index buffer (for reuse)
    pub vertex_count: u32,                        // Number of vertices
    pub index_count: u32,                          // Number of indices (if using)
}

impl Mesh {
    // * Create a new mesh from CPU-side data
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        vertices: &[VertexPosColor],
        indices: Option<&[u32]>,
    ) -> Self {
        // Upload vertices to GPU
        let vertex_buffer = Buffer::from_iter(
            &memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices.iter().copied(),
        ).unwrap();

        let vertex_count = vertices.len() as u32;

        // Upload indices if provided
        let (index_buffer, index_count) = if let Some(indices) = indices {
            let buffer = Buffer::from_iter(
                &memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                indices.iter().copied(),
            ).unwrap();
            (Some(buffer), indices.len() as u32)
        } else {
            (None, 0)
        };

        Self {
            vertices: vertex_buffer,
            indices: index_buffer,
            vertex_count,
            index_count,
        }
    }
}

// ! TRANSFORM - Position, rotation, and scale of an object
#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub position: [f32; 3],  // Translation in world space
    pub rotation: [f32; 3],   // Euler angles (future use)
    pub scale: [f32; 3],      // Scale factors
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],  // Default to no scaling
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
}

// ! SCENE OBJECT - An instance of a mesh in the world
pub struct SceneObject {
    pub mesh: Mesh,
    pub transform: Transform,
    pub instance_data: Option<Subbuffer<[VertexPosColor]>>,  // For instanced rendering (future)
}

impl SceneObject {
    pub fn new(mesh: Mesh) -> Self {
        Self {
            mesh,
            transform: Transform::default(),
            instance_data: None,
        }
    }

    // * Builder pattern methods
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_instance_data(
        mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        instances: &[VertexPosColor],
    ) -> Self {
        let buffer = Buffer::from_iter(
            &memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            instances.iter().copied(),
        ).unwrap();
        
        self.instance_data = Some(buffer);
        self
    }
}

// ! SCENE - Collection of objects to render
pub struct Scene {
    pub objects: Vec<SceneObject>,
}

impl Scene {
    pub fn new() -> Self {
        Self { objects: Vec::new() }
    }

    pub fn add_object(&mut self, object: SceneObject) {
        self.objects.push(object);
    }
}

// ! PRIMITIVE SHAPES - Factory functions for creating common meshes
pub mod shapes {
    use super::*;

    // * Create a unit cube centered at origin
    pub fn create_cube(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let mut vertices: Vec<VertexPosColor> = Vec::new();
        // 8 vertices (one for each corner)
        let mut add_tri = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]| {
            vertices.push(VertexPosColor { position: p0, color, barycentric: [1.0, 0.0, 0.0] });
            vertices.push(VertexPosColor { position: p1, color, barycentric: [0.0, 1.0, 0.0] });
            vertices.push(VertexPosColor { position: p2, color, barycentric: [0.0, 0.0, 1.0] });
        };

        let v = [
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        ];

        add_tri(v[0], v[1], v[2]); add_tri(v[2], v[3], v[0]);
        add_tri(v[1], v[5], v[6]); add_tri(v[6], v[2], v[1]);
        add_tri(v[5], v[4], v[7]); add_tri(v[7], v[6], v[5]);
        add_tri(v[4], v[0], v[3]); add_tri(v[3], v[7], v[4]);
        add_tri(v[3], v[2], v[6]); add_tri(v[6], v[7], v[3]);
        add_tri(v[4], v[5], v[1]); add_tri(v[1], v[0], v[4]);

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create a sphere by subdividing into sectors and stacks
    pub fn create_sphere(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        sectors: u32,
        stacks: u32,
    ) -> Mesh {
        let mut vertices = Vec::new();

        let mut grid = Vec::new();
        for i in 0..=stacks {
            let stack_angle = std::f32::consts::PI / 2.0 - (i as f32) * std::f32::consts::PI / stacks as f32;
            let xy = stack_angle.cos();
            let z = stack_angle.sin();
            for j in 0..=sectors {
                let sector_angle = (j as f32) * 2.0 * std::f32::consts::PI / sectors as f32;
                grid.push([xy * sector_angle.cos(), xy * sector_angle.sin(), z]);
            }
        }

        for i in 0..stacks {
            for j in 0..sectors {
                let p1 = i * (sectors + 1) + j;
                let p2 = p1 + (sectors + 1);
                let p3 = p1 + 1;
                let p4 = p2 + 1;

                vertices.push(VertexPosColor { position: grid[p1 as usize], color, barycentric: [1.0, 0.0, 0.0] });
                vertices.push(VertexPosColor { position: grid[p2 as usize], color, barycentric: [0.0, 1.0, 0.0] });
                vertices.push(VertexPosColor { position: grid[p3 as usize], color, barycentric: [0.0, 0.0, 1.0] });
                vertices.push(VertexPosColor { position: grid[p2 as usize], color, barycentric: [1.0, 0.0, 0.0] });
                vertices.push(VertexPosColor { position: grid[p4 as usize], color, barycentric: [0.0, 1.0, 0.0] });
                vertices.push(VertexPosColor { position: grid[p3 as usize], color, barycentric: [0.0, 0.0, 1.0] });
            }
        }

        Mesh::new(memory_allocator, &vertices, None)
    }

        // * Create a flat plane
        pub fn create_plane(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        width: f32,
        height: f32,
    ) -> Mesh {
        let w = width / 2.0;
        let h = height / 2.0;

        let v0 = [-w, -h, 0.0];
        let v1 = [w, -h, 0.0];
        let v2 = [w, h, 0.0];
        let v3 = [-w, h, 0.0];

        let vertices = vec![
            VertexPosColor { position: v0, color, barycentric: [1.0, 0.0, 0.0] },
            VertexPosColor { position: v1, color, barycentric: [0.0, 1.0, 0.0] },
            VertexPosColor { position: v2, color, barycentric: [0.0, 0.0, 1.0] },
            VertexPosColor { position: v2, color, barycentric: [1.0, 0.0, 0.0] },
            VertexPosColor { position: v3, color, barycentric: [0.0, 1.0, 0.0] },
            VertexPosColor { position: v0, color, barycentric: [0.0, 0.0, 1.0] },
        ];

        Mesh::new(memory_allocator, &vertices, None)
    }
}