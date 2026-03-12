// ! SHAPE AND MESH SYSTEM - Defines geometry and scene organization
// More I do this, more I understand and respect all developers that created a blender, unity, unreal engine and etc. just F for every man that did something like that and biggest F for that people, who did it opensource. F
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

// ! VERTEX WITH NORMALS - For lighting calculations
#[repr(C)]
#[derive(Copy, Clone, Debug, Vertex, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexPosColorNormal {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub barycentric: [f32; 3],
}

impl VertexType for VertexPosColorNormal {}

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

    // * Create a mesh with indexed geometry (more efficient for complex shapes)
    pub fn new_indexed(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        vertices: &[VertexPosColor],
        indices: &[u32],
    ) -> Self {
        Self::new(memory_allocator, vertices, Some(indices))
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

// ! SCENE OBJECT - An instance of a mesh in the world
pub struct SceneObject {
    pub mesh: Mesh,
    pub transform: Transform,
    pub instance_data: Option<Subbuffer<[VertexPosColor]>>,  // For instanced rendering
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

    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.position = [x, y, z];
        self
    }

    pub fn with_rotation(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.rotation = [x, y, z];
        self
    }

    pub fn with_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.scale = [x, y, z];
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

// ? Why am I even writing a comments if I am the only one who read it, am I schizophrenic?

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

    pub fn clear(&mut self) {
        self.objects.clear();
    }
}

// ! PRIMITIVE SHAPES - Factory functions for creating common meshes
pub mod shapes {
    use super::*;
    use std::f32::consts::PI;

    // Helper function to normalize a vector
    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/len, v[1]/len, v[2]/len]
    }

    // Helper function to add a triangle with barycentric coordinates
    fn add_triangle(vertices: &mut Vec<VertexPosColor>, p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], color: [f32; 3]) {
        vertices.push(VertexPosColor { position: p1, color, barycentric: [1.0, 0.0, 0.0] });
        vertices.push(VertexPosColor { position: p2, color, barycentric: [0.0, 1.0, 0.0] });
        vertices.push(VertexPosColor { position: p3, color, barycentric: [0.0, 0.0, 1.0] });
    }

    // Helper function to add a quad as two triangles
    // I dont know who will use it, maybe some chill guy                      or lady....
    fn add_quad(vertices: &mut Vec<VertexPosColor>, p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], p4: [f32; 3], color: [f32; 3]) {
        // First triangle
        vertices.push(VertexPosColor { position: p1, color, barycentric: [1.0, 0.0, 0.0] });
        vertices.push(VertexPosColor { position: p2, color, barycentric: [0.0, 1.0, 0.0] });
        vertices.push(VertexPosColor { position: p3, color, barycentric: [0.0, 0.0, 1.0] });
        
        // Second triangle
        vertices.push(VertexPosColor { position: p3, color, barycentric: [1.0, 0.0, 0.0] });
        vertices.push(VertexPosColor { position: p4, color, barycentric: [0.0, 1.0, 0.0] });
        vertices.push(VertexPosColor { position: p1, color, barycentric: [0.0, 0.0, 1.0] });
    }

    // * Create a unit cube centered at origin
    pub fn create_cube(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let mut vertices: Vec<VertexPosColor> = Vec::new();

        let v = [
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        ];

        // Front face
        add_triangle(&mut vertices, v[0], v[1], v[2], color);
        add_triangle(&mut vertices, v[2], v[3], v[0], color);
        
        // Right face
        add_triangle(&mut vertices, v[1], v[5], v[6], color);
        add_triangle(&mut vertices, v[6], v[2], v[1], color);
        
        // Back face
        add_triangle(&mut vertices, v[5], v[4], v[7], color);
        add_triangle(&mut vertices, v[7], v[6], v[5], color);
        
        // Left face
        add_triangle(&mut vertices, v[4], v[0], v[3], color);
        add_triangle(&mut vertices, v[3], v[7], v[4], color);
        
        // Top face
        add_triangle(&mut vertices, v[3], v[2], v[6], color);
        add_triangle(&mut vertices, v[6], v[7], v[3], color);
        
        // Bottom face
        add_triangle(&mut vertices, v[4], v[5], v[1], color);
        add_triangle(&mut vertices, v[1], v[0], v[4], color);

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
        let radius = 0.5;

        for i in 0..=stacks {
            let stack_angle = PI / 2.0 - (i as f32) * PI / stacks as f32;
            let xy = radius * stack_angle.cos();
            let z = radius * stack_angle.sin();
            
            for j in 0..=sectors {
                let sector_angle = (j as f32) * 2.0 * PI / sectors as f32;
                let x = xy * sector_angle.cos();
                let y = xy * sector_angle.sin();
                
                vertices.push(VertexPosColor { 
                    position: [x, y, z], 
                    color, 
                    barycentric: [1.0, 0.0, 0.0] 
                });
            }
        }

        let mut indices = Vec::new();
        for i in 0..stacks {
            for j in 0..sectors {
                let p1 = i * (sectors + 1) + j;
                let p2 = p1 + (sectors + 1);
                let p3 = p1 + 1;
                let p4 = p2 + 1;

                indices.push(p1);
                indices.push(p2);
                indices.push(p3);
                
                indices.push(p2);
                indices.push(p4);
                indices.push(p3);
            }
        }

        // Reorder vertices based on indices to avoid indexed rendering
        let mut triangle_vertices = Vec::new();
        for i in (0..indices.len()).step_by(3) {
            let v1 = vertices[indices[i] as usize];
            let v2 = vertices[indices[i + 1] as usize];
            let v3 = vertices[indices[i + 2] as usize];
            triangle_vertices.push(v1);
            triangle_vertices.push(v2);
            triangle_vertices.push(v3);
        }

        Mesh::new(memory_allocator, &triangle_vertices, None)
    }

    // * Create a sphere using icosahedron subdivision (better distribution)
    pub fn create_sphere_subdivided(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        subdivisions: u32,
    ) -> Mesh {
        // let mut vertices: Vec<_> = Vec::new();  // * Just will leave it here, bc why not
        let t = (1.0 + (5.0_f32).sqrt()) / 2.0;
        
        let initial_vertices = [
            [-1.0,  t,  0.0], [ 1.0,  t,  0.0], [-1.0, -t,  0.0], [ 1.0, -t,  0.0],
            [ 0.0, -1.0,  t], [ 0.0,  1.0,  t], [ 0.0, -1.0, -t], [ 0.0,  1.0, -t],
            [ t,  0.0, -1.0], [ t,  0.0,  1.0], [-t,  0.0, -1.0], [-t,  0.0,  1.0],
        ];

        let initial_indices = [
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
        ];

        // Normalize all vertices to lie on sphere
        let mut normalized_vertices: Vec<[f32; 3]> = initial_vertices.iter()
            .map(|v| normalize(*v))
            .collect();

        // Subdivide
        let mut indices = initial_indices.to_vec();
        
        for _ in 0..subdivisions {
            let mut new_indices = Vec::new();
            let mut mid_point_cache = std::collections::HashMap::new();
            
            for i in (0..indices.len()).step_by(3) {
                let v1 = normalized_vertices[indices[i] as usize];
                let v2 = normalized_vertices[indices[i + 1] as usize];
                let v3 = normalized_vertices[indices[i + 2] as usize];
                
                // Get or create midpoints
                let a = *mid_point_cache.entry((indices[i], indices[i + 1]))
                    .or_insert_with(|| {
                        normalized_vertices.push(normalize([
                            (v1[0] + v2[0]) * 0.5,
                            (v1[1] + v2[1]) * 0.5,
                            (v1[2] + v2[2]) * 0.5,
                        ]));
                        (normalized_vertices.len() - 1) as u32
                    });
                
                let b = *mid_point_cache.entry((indices[i + 1], indices[i + 2]))
                    .or_insert_with(|| {
                        normalized_vertices.push(normalize([
                            (v2[0] + v3[0]) * 0.5,
                            (v2[1] + v3[1]) * 0.5,
                            (v2[2] + v3[2]) * 0.5,
                        ]));
                        (normalized_vertices.len() - 1) as u32
                    });
                
                let c = *mid_point_cache.entry((indices[i + 2], indices[i]))
                    .or_insert_with(|| {
                        normalized_vertices.push(normalize([
                            (v3[0] + v1[0]) * 0.5,
                            (v3[1] + v1[1]) * 0.5,
                            (v3[2] + v1[2]) * 0.5,
                        ]));
                        (normalized_vertices.len() - 1) as u32
                    });
                
                // Create 4 triangles
                new_indices.extend_from_slice(&[indices[i], a, c]);
                new_indices.extend_from_slice(&[indices[i + 1], b, a]);
                new_indices.extend_from_slice(&[indices[i + 2], c, b]);
                new_indices.extend_from_slice(&[a, b, c]);
            }
            indices = new_indices;
        }

        // Create final vertex list
        let mut final_vertices = Vec::new();
        for i in (0..indices.len()).step_by(3) {
            let v1 = normalized_vertices[indices[i] as usize];
            let v2 = normalized_vertices[indices[i + 1] as usize];
            let v3 = normalized_vertices[indices[i + 2] as usize];
            
            final_vertices.push(VertexPosColor { position: v1, color, barycentric: [1.0, 0.0, 0.0] });
            final_vertices.push(VertexPosColor { position: v2, color, barycentric: [0.0, 1.0, 0.0] });
            final_vertices.push(VertexPosColor { position: v3, color, barycentric: [0.0, 0.0, 1.0] });
        }

        Mesh::new(memory_allocator, &final_vertices, None)
    }

    // * Create a tetrahedron
    pub fn create_tetrahedron(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let mut vertices = Vec::new();
        let a = 0.5;
        
        // Four vertices of a regular tetrahedron
        let v = [
            [ a,  a,  a],
            [ a, -a, -a],
            [-a,  a, -a],
            [-a, -a,  a],
        ];

        // Four faces (each triangle)
        add_triangle(&mut vertices, v[0], v[1], v[2], color);
        add_triangle(&mut vertices, v[0], v[2], v[3], color);
        add_triangle(&mut vertices, v[0], v[3], v[1], color);
        add_triangle(&mut vertices, v[1], v[3], v[2], color);

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create an octahedron
    pub fn create_octahedron(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let mut vertices = Vec::new();
        let a = 0.5;
        
        // Six vertices
        let v = [
            [ a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0,  a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0,  a],
            [0.0, 0.0, -a],
        ];

        // Top half (4 triangles)
        add_triangle(&mut vertices, v[4], v[0], v[2], color);
        add_triangle(&mut vertices, v[4], v[2], v[1], color);
        add_triangle(&mut vertices, v[4], v[1], v[3], color);
        add_triangle(&mut vertices, v[4], v[3], v[0], color);
        
        // Bottom half (4 triangles)
        add_triangle(&mut vertices, v[5], v[2], v[0], color);
        add_triangle(&mut vertices, v[5], v[1], v[2], color);
        add_triangle(&mut vertices, v[5], v[3], v[1], color);
        add_triangle(&mut vertices, v[5], v[0], v[3], color);

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create a dodecahedron
    pub fn create_dodecahedron(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let mut vertices = Vec::new();
        let phi = (1.0 + (5.0_f32).sqrt()) / 2.0; // Golden ratio
        let a = 0.3;
        let b = a / phi;
        let c = a * phi;

        // 20 vertices
        let v = [
            [ a,  a,  a], [ a,  a, -a], [ a, -a,  a], [ a, -a, -a],
            [-a,  a,  a], [-a,  a, -a], [-a, -a,  a], [-a, -a, -a],
            [ 0.0,  b, -c], [ 0.0, -b, -c], [ 0.0,  b,  c], [ 0.0, -b,  c],
            [ b,  c,  0.0], [ b, -c,  0.0], [-b,  c,  0.0], [-b, -c,  0.0],
            [ c,  0.0,  b], [ c,  0.0, -b], [-c,  0.0,  b], [-c,  0.0, -b],
        ];

        // 12 pentagonal faces (each split into 3 triangles)
        let faces = [
            [0, 10, 11, 2, 16], [0, 16, 17, 1, 8], [0, 8, 4, 14, 10],
            [1, 17, 3, 9, 5], [1, 5, 13, 12, 8], [2, 11, 6, 18, 15],
            [2, 15, 13, 3, 16], [3, 13, 5, 9, 17], [4, 8, 12, 14, 18],
            [4, 18, 6, 11, 10], [5, 9, 7, 19, 13], [6, 15, 19, 7, 18],
        ];

        for face in faces.iter() {
            // Triangulate pentagon (fan from first vertex)
            for i in 1..(face.len() - 1) {
                add_triangle(&mut vertices, v[face[0]], v[face[i]], v[face[i + 1]], color);
            }
        }

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create an icosahedron
    pub fn create_icosahedron(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
    ) -> Mesh {
        let mut vertices = Vec::new();
        let phi = (1.0 + (5.0_f32).sqrt()) / 2.0;
        let a = 0.5;

        // 12 vertices
        let v = [
            [-a,  phi,  0.0], [ a,  phi,  0.0], [-a, -phi,  0.0], [ a, -phi,  0.0],
            [0.0, -a,  phi], [0.0,  a,  phi], [0.0, -a, -phi], [0.0,  a, -phi],
            [ phi,  0.0, -a], [ phi,  0.0,  a], [-phi,  0.0, -a], [-phi,  0.0,  a],
        ];

        // 20 triangular faces
        let faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];

        for face in faces.iter() {
            add_triangle(&mut vertices, v[face[0]], v[face[1]], v[face[2]], color);
        }

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create a torus
    pub fn create_torus(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
    ) -> Mesh {
        let mut vertices = Vec::new();

        for i in 0..major_segments {
            let major_angle = (i as f32) * 2.0 * PI / major_segments as f32;
            let next_major_angle = ((i + 1) as f32) * 2.0 * PI / major_segments as f32;
            
            let cos_major = major_angle.cos();
            let sin_major = major_angle.sin();
            let cos_next_major = next_major_angle.cos();
            let sin_next_major = next_major_angle.sin();

            for j in 0..minor_segments {
                let minor_angle = (j as f32) * 2.0 * PI / minor_segments as f32;
                let next_minor_angle = ((j + 1) as f32) * 2.0 * PI / minor_segments as f32;
                
                let cos_minor = minor_angle.cos();
                let sin_minor = minor_angle.sin();
                let cos_next_minor = next_minor_angle.cos();
                let sin_next_minor = next_minor_angle.sin();

                // Calculate the four corners of the quad
                let p1 = [
                    (major_radius + minor_radius * cos_minor) * cos_major,
                    (major_radius + minor_radius * cos_minor) * sin_major,
                    minor_radius * sin_minor,
                ];
                
                let p2 = [
                    (major_radius + minor_radius * cos_next_minor) * cos_major,
                    (major_radius + minor_radius * cos_next_minor) * sin_major,
                    minor_radius * sin_next_minor,
                ];
                
                let p3 = [
                    (major_radius + minor_radius * cos_next_minor) * cos_next_major,
                    (major_radius + minor_radius * cos_next_minor) * sin_next_major,
                    minor_radius * sin_next_minor,
                ];
                
                let p4 = [
                    (major_radius + minor_radius * cos_minor) * cos_next_major,
                    (major_radius + minor_radius * cos_minor) * sin_next_major,
                    minor_radius * sin_minor,
                ];

                // Create two triangles for the quad
                add_triangle(&mut vertices, p1, p2, p3, color);
                add_triangle(&mut vertices, p3, p4, p1, color);
            }
        }

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create a cylinder
    pub fn create_cylinder(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        radius: f32,
        height: f32,
        sectors: u32,
    ) -> Mesh {
        let mut vertices = Vec::new();
        let half_height = height / 2.0;

        // Create vertices around the circle
        let mut circle_points = Vec::new();
        for i in 0..sectors {
            let angle = (i as f32) * 2.0 * PI / sectors as f32;
            circle_points.push([
                radius * angle.cos(),
                radius * angle.sin(),
            ]);
        }

        // Side faces (quads)
        for i in 0..sectors {
            // Convert to usize for indexing
            let current_idx = i as usize;
            let next_idx = ((i + 1) % sectors) as usize;
            
            let p1 = [
                circle_points[current_idx][0], 
                circle_points[current_idx][1], 
                -half_height
            ];
            let p2 = [
                circle_points[next_idx][0], 
                circle_points[next_idx][1], 
                -half_height
            ];
            let p3 = [
                circle_points[next_idx][0], 
                circle_points[next_idx][1], 
                half_height
            ];
            let p4 = [
                circle_points[current_idx][0], 
                circle_points[current_idx][1], 
                half_height
            ];

            add_triangle(&mut vertices, p1, p2, p3, color);
            add_triangle(&mut vertices, p3, p4, p1, color);
        }

        // Top cap (triangles from center)
        for i in 0..sectors {
            let current_idx = i as usize;
            let next_i = ((i + 1) % sectors) as usize;
            let p1 = [0.0, 0.0, half_height];
            let p2 = [circle_points[current_idx][0], circle_points[current_idx][1], half_height];
            let p3 = [circle_points[next_i][0], circle_points[next_i][1], half_height];

            add_triangle(&mut vertices, p1, p2, p3, color);
        }

        // Bottom cap (triangles from center)
        for i in 0..sectors {
            let current_idx = i as usize;
            let next_i = ((i + 1) % sectors) as usize;
            
            let p1 = [0.0, 0.0, -half_height];
            let p2 = [circle_points[current_idx][0], circle_points[current_idx][1], -half_height];
            let p3 = [circle_points[next_i][0], circle_points[next_i][1], -half_height];

            add_triangle(&mut vertices, p1, p2, p3, color);
        }

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create a cone
    pub fn create_cone(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        radius: f32,
        height: f32,
        sectors: u32,
    ) -> Mesh {
        let mut vertices = Vec::new();
        let half_height = height / 2.0;

        // Create vertices around the circle
        let mut circle_points = Vec::new();
        for i in 0..sectors {
            let angle = (i as f32) * 2.0 * PI / sectors as f32;
            circle_points.push([
                radius * angle.cos(),
                radius * angle.sin(),
            ]);
        }

        // Tip of cone
        let tip = [0.0, 0.0, half_height];

        // Side faces (triangles)
        for i in 0..sectors {
            let current_idx = i as usize;
            let next_i = ((i + 1) % sectors) as usize;
            let p1 = tip;
            let p2 = [circle_points[current_idx][0], circle_points[current_idx][1], -half_height];
            let p3 = [circle_points[next_i][0], circle_points[next_i][1], -half_height];

            add_triangle(&mut vertices, p1, p2, p3, color);
        }

        // Bottom cap (triangles from center)
        for i in 0..sectors {
            let current_idx = i as usize;
            let next_i = ((i + 1) % sectors) as usize;
            let p1 = [0.0, 0.0, -half_height];
            let p2 = [circle_points[current_idx][0], circle_points[current_idx][1], -half_height];
            let p3 = [circle_points[next_i][0], circle_points[next_i][1], -half_height];

            add_triangle(&mut vertices, p1, p2, p3, color);
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

    // * Create a grid of lines (useful for debugging/visualization)
    // Now its looks shit, but I will make it cool, dont worry
    pub fn create_grid(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        size: f32,
        divisions: u32,
    ) -> Mesh {
        let mut vertices = Vec::new();
        let step = size / divisions as f32;
        let half = size / 2.0;

        // Lines along X axis
        for i in 0..=divisions {
            let x = -half + i as f32 * step;
            vertices.push(VertexPosColor { position: [x, -half, 0.0], color, barycentric: [1.0, 0.0, 0.0] });
            vertices.push(VertexPosColor { position: [x, half, 0.0], color, barycentric: [0.0, 1.0, 0.0] });
        }

        // Lines along Y axis
        for i in 0..=divisions {
            let y = -half + i as f32 * step;
            vertices.push(VertexPosColor { position: [-half, y, 0.0], color, barycentric: [0.0, 0.0, 1.0] });
            vertices.push(VertexPosColor { position: [half, y, 0.0], color, barycentric: [1.0, 0.0, 0.0] });
        }

        Mesh::new(memory_allocator, &vertices, None)
    }

    // * Create a pyramid (square base)
    pub fn create_pyramid(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color: [f32; 3],
        size: f32,
        height: f32,
    ) -> Mesh {
        let mut vertices = Vec::new();
        let half = size / 2.0;
        let half_height = height / 2.0;

        // Base(after base) vertices
        let b1 = [-half, -half, -half_height];
        let b2 = [half, -half, -half_height];
        let b3 = [half, half, -half_height];
        let b4 = [-half, half, -half_height];
        
        // Apex, like a game
        let apex = [0.0, 0.0, half_height];

        // Base (two triangles)
        add_triangle(&mut vertices, b1, b2, b3, color);
        add_triangle(&mut vertices, b3, b4, b1, color);

        // Four sides
        add_triangle(&mut vertices, apex, b1, b2, color);
        add_triangle(&mut vertices, apex, b2, b3, color);
        add_triangle(&mut vertices, apex, b3, b4, color);
        add_triangle(&mut vertices, apex, b4, b1, color);

        Mesh::new(memory_allocator, &vertices, None)
    }
}