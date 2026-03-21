pub mod animation;
pub mod object;

use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::ImmutableImage;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::sync::GpuFuture;
use rayon::prelude::*;

use crate::scene::object::{RenderBatch, Instance, InstanceData, Texture};
use crate::scene::animation::AnimationType;
use crate::renderer::pipeline::UniformBufferObject;
use crate::shapes::Mesh;
use crate::input::MouseState;

pub struct RenderScene {
    pub batches: Vec<RenderBatch>,
    pub frames: Vec<FrameData>,
    pub light_pos: [f32; 3],
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub texture_views: Vec<Arc<ImageView<ImmutableImage>>>,
    pub texture_sampler: Arc<Sampler>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

pub struct FrameData {
    pub instance_buffer: Subbuffer<[InstanceData]>,
    pub uniform_buffer: Subbuffer<UniformBufferObject>,
}

#[derive(Clone, Copy)]
pub struct InstanceHandle {
    pub batch_index: usize,
    pub instance_index: usize,
}

impl RenderScene {

    pub fn add_instance(&mut self, mesh: Mesh, instance: Instance) -> InstanceHandle {
        for (batch_index, batch) in self.batches.iter_mut().enumerate() {
            if batch.mesh.vertices.buffer() == mesh.vertices.buffer() {
                batch.instances.push(instance);

                return InstanceHandle {
                    batch_index,
                    instance_index: batch.instances.len() - 1,
                };
            }
        }

        self.batches.push(RenderBatch {
            mesh,
            instances: vec![instance],
        });

        InstanceHandle {
            batch_index: self.batches.len() - 1,
            instance_index: 0,
        }
    }

    pub fn remove_instance(&mut self, handle: InstanceHandle) {
        if let Some(batch) = self.batches.get_mut(handle.batch_index) {
            if handle.instance_index < batch.instances.len() {
                batch.instances.swap_remove(handle.instance_index);
            }

            // Optional: remove empty batches
            if batch.instances.is_empty() {
                self.batches.swap_remove(handle.batch_index);
            }
        }
    }

    // ! This is important part, the 'max_instances' is not so cool, I will try to rewrite it 
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        _pipeline: &Arc<GraphicsPipeline>,
        queue: &Arc<Queue>,
        frames_in_flight: usize,
        max_instances: usize
    ) -> Self {
        let mut frames = Vec::new();
        let default_texture = Self::create_texture_image(memory_allocator, queue, &[255, 255, 255, 255], 1, 1);
        let default_texture_view = ImageView::new_default(default_texture).unwrap();
        let texture_sampler = Sampler::new(
            queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        ).unwrap();

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
    
            frames.push(FrameData {
                instance_buffer,
                uniform_buffer,
            });
        }

        Self {
            batches: Vec::new(),
            frames,
            light_pos: [0.0, 10.0, 0.0], 
            light_color: [1.0, 1.0, 1.0],
            light_intensity: 50.0,
            texture_views: vec![default_texture_view],
            texture_sampler,
            descriptor_set_allocator: descriptor_set_allocator.clone(),
        }
    }

    fn create_texture_image(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        queue: &Arc<Queue>,
        pixels_rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Arc<ImmutableImage> {
        let cb_allocator = StandardCommandBufferAllocator::new(
            queue.device().clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );
        let mut upload_builder = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();
        let image = ImmutableImage::from_iter::<u8, _, _, _>(
            memory_allocator.as_ref(),
            pixels_rgba.iter().copied(),
            vulkano::image::ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1,
            },
            vulkano::image::MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            &mut upload_builder,
        ).unwrap();
        let upload_cmd = upload_builder.build().unwrap();
        vulkano::sync::now(queue.device().clone())
            .then_execute(queue.clone(), upload_cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        image
    }

    fn to_rgba8(tex: &Texture) -> Vec<u8> {
        match tex.pixels.len() as u32 {
            len if len == tex.width * tex.height * 4 => tex.pixels.clone(),
            len if len == tex.width * tex.height * 3 => {
                let mut out = Vec::with_capacity((tex.width * tex.height * 4) as usize);
                for rgb in tex.pixels.chunks_exact(3) {
                    out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                }
                out
            }
            _ => vec![255, 255, 255, 255],
        }
    }

    pub fn set_textures(
        &mut self,
        textures: &[Texture],
        queue: &Arc<Queue>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) {
        for tex in textures {
            if tex.width == 0 || tex.height == 0 {
                continue;
            }
            let pixels_rgba = Self::to_rgba8(tex);
            let image = Self::create_texture_image(memory_allocator, queue, &pixels_rgba, tex.width, tex.height);
            let view = ImageView::new_default(image).unwrap();
            self.texture_views.push(view);
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
                        logic(&mut inst.transform, &mut inst.velocity, &mut inst.original_position, &mut inst.color, elapsed);
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
        eye_pos: [f32; 3],
    ) {
        let frame = &self.frames[frame_index];
        
        let total_instances: usize = self.batches.iter().map(|b| b.instances.len()).sum();
        if total_instances == 0 {
            return; 
        }
        // println!("View: {:?}, eye_pos: {:?}", view, eye_pos);
        assert!(total_instances <= frame.instance_buffer.len() as usize);
        {
            let mut ubo_data = frame.uniform_buffer.write().unwrap();
            ubo_data.view = view;
            ubo_data.proj = proj;
            ubo_data.eye_pos = eye_pos;
            
            ubo_data.light_pos = self.light_pos;
            ubo_data.light_color = self.light_color;
            ubo_data.light_intensity = self.light_intensity;
        }
        // Map the buffer and write data
        {
            let mut data = frame.instance_buffer.write().unwrap();
            let mut current_instance = 0;
            
            for batch in &self.batches {
                for inst in &batch.instances {
                    data[current_instance] = InstanceData { 
                        model: inst.model_matrix, 
                        color: inst.color, 
                        padding: 0.0, 
                        mat_props: [inst.roughness, inst.metalness, 0.0, 0.0],
                 };
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
            
            let texture_index = batch
                .mesh
                .base_color_texture
                .filter(|idx| *idx < self.texture_views.len())
                .unwrap_or(0);
            let descriptor_set = PersistentDescriptorSet::new(
                self.descriptor_set_allocator.as_ref(),
                pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, frame.uniform_buffer.clone()),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        self.texture_views[texture_index].clone(),
                        self.texture_sampler.clone(),
                    ),
                ],
            ).unwrap();
            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
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

    pub fn set_light(&mut self, position: [f32; 3], color: [f32; 3], intensity: f32) {
        self.light_pos = position;
        self.light_color = color;
        self.light_intensity = intensity;
    }
}