use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::command_buffer::AutoCommandBufferBuilder;

pub trait VertexType: Vertex + bytemuck::Pod + bytemuck::Zeroable + Send + Sync {}

#[repr(C)]
#[derive(Copy, Clone, Debug, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexPosColor {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

impl VertexType for VertexPosColor {}

#[repr(C)]
#[derive(Copy, Clone, Debug, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexPosColorUv {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

impl VertexType for VertexPosColorUv {}

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Subbuffer<[VertexPosColor]>,  
    pub indices: Option<Subbuffer<[u32]>>,
    pub vertex_count: u32,
    pub index_count: u32,
}

impl Mesh {
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        vertices: &[VertexPosColor],
        indices: Option<&[u32]>,
    ) -> Self {
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

#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 3], 
    pub scale: [f32; 3],
}

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
        let (sx, sy, sz) = (self.scale[0], self.scale[1], self.scale[2]);
        let (tx, ty, tz) = (self.position[0], self.position[1], self.position[2]);
        
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [tx, ty, tz, 1.0],
        ]
    }
}

pub struct SceneObject {
    pub mesh: Mesh,
    pub transform: Transform,
    pub instance_data: Option<Subbuffer<[VertexPosColor]>>,
}

impl SceneObject {
    pub fn new(mesh: Mesh) -> Self {
        Self {
            mesh,
            transform: Transform::default(),
            instance_data: None,
        }
    }

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

pub mod shapes {
    use super::*;

    pub fn create_cube(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let vertices = vec![
            // front face
            VertexPosColor { position: [-0.5, -0.5, 0.5], color },
            VertexPosColor { position: [0.5, -0.5, 0.5], color },
            VertexPosColor { position: [0.5, 0.5, 0.5], color },
            VertexPosColor { position: [-0.5, 0.5, 0.5], color },
            // back face
            VertexPosColor { position: [-0.5, -0.5, -0.5], color },
            VertexPosColor { position: [0.5, -0.5, -0.5], color },
            VertexPosColor { position: [0.5, 0.5, -0.5], color },
            VertexPosColor { position: [-0.5, 0.5, -0.5], color },
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0, // Front
            1, 5, 6, 6, 2, 1, // Right
            5, 4, 7, 7, 6, 5, // Back
            4, 0, 3, 3, 7, 4, // Left
            3, 2, 6, 6, 7, 3, // Top
            4, 5, 1, 1, 0, 4, // Bottom
        ];

        Mesh::new(memory_allocator, &vertices, Some(&indices))
    }

    pub fn create_sphere(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        sectors: u32,
        stacks: u32,
    ) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for i in 0..=stacks {
            let stack_angle = std::f32::consts::PI / 2.0 - (i as f32) * std::f32::consts::PI / stacks as f32;
            let xy = stack_angle.cos();
            let z = stack_angle.sin();

            for j in 0..=sectors {
                let sector_angle = (j as f32) * 2.0 * std::f32::consts::PI / sectors as f32;
                let x = xy * sector_angle.cos();
                let y = xy * sector_angle.sin();

                vertices.push(VertexPosColor {
                    position: [x, y, z],
                    color,
                });
            }
        }

        for i in 0..stacks {
            for j in 0..sectors {
                let first = i * (sectors + 1) + j;
                let second = first + sectors + 1;

                indices.push(first);
                indices.push(second);
                indices.push(first + 1);

                indices.push(second);
                indices.push(second + 1);
                indices.push(first + 1);
            }
        }

        Mesh::new(memory_allocator, &vertices, Some(&indices))
    }

    pub fn create_plane(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        width: f32,
        height: f32,
    ) -> Mesh {
        let w = width / 2.0;
        let h = height / 2.0;

        let vertices = vec![
            VertexPosColor { position: [-w, -h, 0.0], color },
            VertexPosColor { position: [w, -h, 0.0], color },
            VertexPosColor { position: [w, h, 0.0], color },
            VertexPosColor { position: [-w, h, 0.0], color },
        ];

        let indices = vec![0, 1, 2, 2, 3, 0];
        Mesh::new(memory_allocator, &vertices, Some(&indices))
    }
}