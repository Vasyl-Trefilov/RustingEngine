// So, here we go, a start of my library, a Scene
use crate::shapes::*;
use crate::{RenderObject, create_render_object, constrain_to_screen};
use vulkano::device::Device;
use vulkano::memory::allocator::{MemoryAllocator, StandardMemoryAllocator};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use std::sync::Arc;
use crate::MouseState;
use crate::AnimationType;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::Pipeline;
use crate::{apply_mouse_repulsion, RenderBatch, Instance, InstanceData};
use vulkano::buffer::Buffer;
use vulkano::memory::allocator::GenericMemoryAllocator;
use vulkano::memory::allocator::FreeListAllocator;
use crate::BufferCreateInfo;
use crate::BufferUsage;
use crate::AllocationCreateInfo;
use vulkano::buffer::Subbuffer;
use crate::UniformBufferObject;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::MemoryUsage;
use rayon::prelude::*; 

pub struct RenderScene {
    batches: Vec<RenderBatch>,
    frames: Vec<FrameData>,
}

// I use FrameData, to avoid GPU and CPU conflict
struct FrameData {
    pub instance_buffer: Subbuffer<[InstanceData]>,
    pub uniform_buffer: Subbuffer<UniformBufferObject>,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
}

impl RenderScene {

    pub fn add_instance(&mut self, mesh: Mesh, instance: Instance) {
        // println!("Adding instance at position: {:?}", instance.transform.position);
        
        for batch in &mut self.batches {
            if batch.mesh.vertices.buffer() == mesh.vertices.buffer() {
                batch.instances.push(instance);
                // println!("Added to existing batch, total instances: {}", batch.instances.len());
                return;
            }
        }
        
        // println!("Creating new batch for mesh");
        self.batches.push(RenderBatch {
            mesh,
            instances: vec![instance],
        });
    }

    // ! This is important part, the 'max_instances' is not so cool, I will try to rewrite it 
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        pipeline: &Arc<GraphicsPipeline>,
        frames_in_flight: usize,
        max_instances: usize
    ) -> Self {
        let mut frames = Vec::new();

        for _ in 0..frames_in_flight {
            let instance_buffer = Buffer::new_slice::<InstanceData>(
                &memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload, 
                    ..Default::default()
                },
                max_instances as u64
            ).unwrap();

            let uniform_buffer = Buffer::from_data(
                &memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                UniformBufferObject::default()
            ).unwrap();
    
            let descriptor_set = PersistentDescriptorSet::new(
                descriptor_set_allocator,
                pipeline.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer.clone())]
            ).unwrap();
            frames.push(FrameData {
                instance_buffer,
                uniform_buffer,
                descriptor_set,
            });
        }

        Self {
            batches: Vec::new(),
            frames,
        }
    }

    // * old version
        // pub fn add_object(&mut self, object: RenderObject) {
    //     self.render_objects.push(object);
    // }
    

    // ! This is update of all object in scene, NOT RENDER, update data of each object
    pub fn update(&mut self, elapsed: f32, mouse: &MouseState) {
        // let aspect = 1920.0 / 1080.0;
        // let world_h = 10.0 * (45.0f32.to_radians() / 2.0).tan();
        // let world_w = world_h * aspect;
        
        // let margin = 1.0;
        // let x_limit = world_w * margin;
        // let y_limit = world_h * margin;

        for batch in &mut self.batches {
            batch.instances.par_iter_mut().for_each(|inst| {
                match &inst.animation {
                    AnimationType::Custom(logic) => {
                        logic(&mut inst.transform, &mut inst.velocity, &mut inst.original_position, elapsed);
                    }
                    AnimationType::Rotate => {
                        inst.transform.rotation[0] = elapsed * 0.2;
                        inst.transform.rotation[1] = elapsed * 0.2;
                    }
                    _ => {}
                }

                // Physics update, now its little, but I will make it bigger
                // inst.transform.position[0] += inst.velocity[0];
                // inst.transform.position[1] += inst.velocity[1];
                // inst.transform.position[2] += inst.velocity[2];
                
                // Calculate the matrix here so Render just a copy, its like +33% performance boost, I cheked
                inst.model_matrix = inst.transform.to_matrix(); 
            });
        }
    }

    // ! This is render, it only draws the object, no data update
    pub fn render(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        _memory_allocator: &Arc<StandardMemoryAllocator>, 
        frame_index: usize,
        view: [[f32; 4]; 4], 
        proj: [[f32; 4]; 4],
    ) {
        let frame = &self.frames[frame_index];
        
        let total_instances: usize = self.batches.iter().map(|b| b.instances.len()).sum();
        if total_instances == 0 {
            return; 
        }
        
        assert!(total_instances <= frame.instance_buffer.len() as usize);
        {
            let mut ubo_data = frame.uniform_buffer.write().unwrap();
            ubo_data.view = view;
            ubo_data.proj = proj;
        }
        // Map the buffer and write data
        {
            let mut data = frame.instance_buffer.write().unwrap();
            let mut current_instance = 0;
            
            for batch in &self.batches {
                for inst in &batch.instances {
                    data[current_instance] = InstanceData { model: inst.model_matrix, color: inst.color, padding: 0.0, };
                    current_instance += 1;
                }
            }
        } // 'data' is dropped here, unlocking the buffer for the GPU

        // now perform the draw calls
        let mut current_instance_offset = 0;
        for batch in &self.batches {
            if batch.instances.is_empty() {
                continue;
            }
            
            builder.bind_vertex_buffers(0, (
                batch.mesh.vertices.clone(),
                frame.instance_buffer.clone(),
            ));
            
            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                frame.descriptor_set.clone(),
            );
            
            if let Some(indices) = &batch.mesh.indices {
                builder.bind_index_buffer(indices.clone());
                builder.draw_indexed(
                    batch.mesh.index_count,
                    batch.instances.len() as u32,
                    0,
                    0,
                    current_instance_offset as u32, 
                ).unwrap();
            } else {
                builder.draw(
                    batch.mesh.vertex_count,
                    batch.instances.len() as u32,
                    0,
                    current_instance_offset as u32,
                ).unwrap();
            }
            
            current_instance_offset += batch.instances.len();
        }
    }
}