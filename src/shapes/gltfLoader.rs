use vulkano::memory::allocator::StandardMemoryAllocator;
use crate::shapes::Mesh;
use std::sync::Arc;
use crate::VertexPosColorNormal;

// * this not how gltf must work

pub fn load_gltf_mesh(
    allocator: &Arc<StandardMemoryAllocator>,
    path: &str,
) -> Mesh {
    let (document, buffers, _) = gltf::import(path).unwrap();

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> =
                reader.read_positions().unwrap().collect();

            let normals: Vec<[f32; 3]> =
                reader.read_normals()
                    .map(|n| n.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            if let Some(read_indices) = reader.read_indices() {
                indices = read_indices.into_u32().collect();
            }

            for i in 0..positions.len() {
                vertices.push(VertexPosColorNormal {
                    position: positions[i],
                    normal: normals[i],
                    color: [1.0, 1.0, 1.0], 
                    barycentric: [0.0, 0.0, 0.0],
                });
            }
        }
    }

    if indices.is_empty() {
        Mesh::new(allocator, &vertices, None)
    } else {
        Mesh::new_indexed(allocator, &vertices, &indices)
    }
}

use crate::Instance;

pub fn load_gltf_scene(
    allocator: &Arc<StandardMemoryAllocator>,
    path: &str,
) -> Vec<(Mesh, Instance)> {
    let (document, buffers, _) = gltf::import(path).unwrap();

    let mut result = Vec::new();

    for scene in document.scenes() {
        for node in scene.nodes() {
            process_node(
                allocator,
                &buffers,
                &mut result,
                &node,
                nalgebra::Matrix4::identity(),
            );
        }
    }

    result
}

use crate::Transform;

fn process_node(
    allocator: &Arc<StandardMemoryAllocator>,
    buffers: &Vec<gltf::buffer::Data>,
    result: &mut Vec<(Mesh, Instance)>,
    node: &gltf::Node,
    parent_transform: nalgebra::Matrix4<f32>,
) {
    let local_transform = nalgebra::Matrix4::from(node.transform().matrix());
    let global_transform = parent_transform * local_transform;

    // If node has a mesh → extract it
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let (vertices, indices) = extract_primitive(&primitive, buffers);

            let mesh = if let Some(ref idx) = indices {
                Mesh::new_indexed(allocator, &vertices, idx)
            } else {
                Mesh::new(allocator, &vertices, None)
            };

            let material = primitive.material();
			let pbr = material.pbr_metallic_roughness();

			// Base color (RGBA)
			let base_color = pbr.base_color_factor();

			// Convert to RGB
			let color = [base_color[0], base_color[1], base_color[2]];

			// Metallic / roughness
			let metalness = pbr.metallic_factor();
			let roughness = pbr.roughness_factor();

			// Optional: derive specular/shininess (your model)
			let specular_strength = 1.0 - roughness;
			let shininess = (1.0 - roughness).powf(4.0) * 128.0;

			let instance = Instance {
				transform: Transform::from_matrix(global_transform),
				color,
				roughness,
				metalness,
				specular_strength,
				shininess,
				..Default::default()
			};

            result.push((mesh, instance));
        }
    }

    // Recurse into children
    for child in node.children() {
        process_node(
            allocator,
            buffers,
            result,
            &child,
            global_transform,
        );
    }
}

fn extract_primitive(
    primitive: &gltf::Primitive,
    buffers: &Vec<gltf::buffer::Data>,
) -> (Vec<VertexPosColorNormal>, Option<Vec<u32>>) {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> =
        reader.read_positions().unwrap().collect();

    let normals: Vec<[f32; 3]> =
        reader.read_normals()
            .map(|n| n.collect())
            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

    let indices = reader
        .read_indices()
        .map(|i| i.into_u32().collect::<Vec<u32>>());

    let mut vertices = Vec::with_capacity(positions.len());

    for i in 0..positions.len() {
        vertices.push(VertexPosColorNormal {
            position: positions[i],
            normal: normals[i],
            color: [1.0, 1.0, 1.0],
            barycentric: [0.0, 0.0, 0.0],
        });
    }

    (vertices, indices)
}
