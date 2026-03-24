pub mod animation;
pub mod object;

use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyBufferInfoTyped, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::ImmutableImage;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::ComputePipeline;
use crate::scene::object::PhysicsPushConstants;
use vulkano::render_pass::Framebuffer;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::DeviceSize;
use rayon::prelude::*;

use crate::input::MouseState;
// use crate::renderer::frustum::{frustum_planes, sphere_visible, view_proj_matrix};
use crate::renderer::pipeline::UniformBufferObject;
use crate::scene::animation::AnimationType;
use crate::scene::object::{Instance, InstanceData, RenderBatch, Texture};
use crate::shapes::Mesh;

pub struct RenderScene {
    // pub max_instances: usize,
    pub batches: Vec<RenderBatch>,
    pub frames: Vec<FrameData>,
    pub light_pos: [f32; 3],
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub texture_views: Vec<Arc<ImageView<ImmutableImage>>>,
    pub texture_sampler: Arc<Sampler>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub descriptor_sets: Vec<Vec<Arc<PersistentDescriptorSet>>>,
    pub physics_read: Subbuffer<[InstanceData]>,
    pub physics_write: Subbuffer<[InstanceData]>,
    pub total_instances: u32,
}

pub struct FrameData {
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

            if batch.instances.is_empty() {
                self.batches.swap_remove(handle.batch_index);
            }
        }
    }

    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        pipeline: &Arc<GraphicsPipeline>,
        queue: &Arc<Queue>,
        frames_in_flight: usize,
        max_instances: usize,
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
                lod: 0.0..=vulkano::sampler::LOD_CLAMP_NONE,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let physics_read = Buffer::new_slice::<InstanceData>(  
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            max_instances as u64, ).unwrap();
        let physics_write = Buffer::new_slice::<InstanceData>(  
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC, 
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            max_instances as u64, ).unwrap();

        for _ in 0..frames_in_flight {
            let uniform_buffer = Buffer::from_data(
                memory_allocator,
                BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
                AllocationCreateInfo { usage: MemoryUsage::Upload, ..Default::default() },
                UniformBufferObject::default(),
            ).unwrap();

            frames.push(FrameData { uniform_buffer });
        }

        let mut scene = Self {
            batches: Vec::new(),
            frames,
            light_pos:[0.0, 10.0, 0.0], light_color: [1.0, 1.0, 1.0], light_intensity: 50.0,
            texture_views: vec![/* default view */],
            texture_sampler, 
            descriptor_set_allocator: descriptor_set_allocator.clone(),
            descriptor_sets: vec![Vec::new(); frames_in_flight],
            // physics_buffer,
            total_instances: 0,
            physics_read,
            physics_write,
        };
        scene
    }

    pub fn upload_to_gpu(&mut self, allocator: &Arc<StandardMemoryAllocator>, queue: &Arc<Queue>) {
        let mut flat_data = Vec::new();
        for batch in &self.batches {
            for inst in &batch.instances {
                flat_data.push(InstanceData {
                    model: inst.model_matrix,
                    color: [inst.color[0], inst.color[1], inst.color[2], inst.emissive],
                    mat_props:[inst.roughness, inst.metalness, 0.0, 0.0],
                    velocity: inst.velocity,
                });
            }
        }
        self.total_instances = flat_data.len() as u32;
        if self.total_instances == 0 { return; }

        let staging = Buffer::from_iter(
            allocator,
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
            AllocationCreateInfo { usage: MemoryUsage::Upload, ..Default::default() },
            flat_data,
        ).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &StandardCommandBufferAllocator::new(queue.device().clone(), Default::default()),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        builder.copy_buffer(CopyBufferInfoTyped::buffers(
            staging.clone(),
            self.physics_read.clone().slice(0..self.total_instances as u64)
        )).unwrap();builder.copy_buffer(CopyBufferInfoTyped::buffers(
            staging,
            self.physics_write.clone().slice(0..self.total_instances as u64)
        )).unwrap();

        let cmd = builder.build().unwrap();
        vulkano::sync::now(queue.device().clone())
            .then_execute(queue.clone(), cmd).unwrap()
            .then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();
    }

    pub fn ensure_descriptor_cache(&mut self, pipeline: &Arc<GraphicsPipeline>, target_count: usize) {
        let layout = pipeline.layout().set_layouts()[0].clone();
        for (frame_i, frame) in self.frames.iter().enumerate() {
            while self.descriptor_sets[frame_i].len() < target_count {
                let tex_i = self.descriptor_sets[frame_i].len();
                let set = PersistentDescriptorSet::new(
                    self.descriptor_set_allocator.as_ref(),
                    layout.clone(),[
                        WriteDescriptorSet::buffer(0, frame.uniform_buffer.clone()), 
                        WriteDescriptorSet::image_view_sampler(1, self.texture_views[tex_i].clone(), self.texture_sampler.clone()),
                        WriteDescriptorSet::buffer(2, self.physics_read.clone())

                    ],
                ).unwrap();
                self.descriptor_sets[frame_i].push(set);
            }
        }
    }

    pub fn prepare_frame_ubo(&mut self, frame_index: usize, view: [[f32; 4]; 4], proj: [[f32; 4]; 4], eye_pos:[f32; 3]) {
        let mut ubo = self.frames[frame_index].uniform_buffer.write().unwrap();
        ubo.view = view; ubo.proj = proj; ubo.eye_pos = eye_pos;
        ubo.light_pos = self.light_pos; ubo.light_color = self.light_color; ubo.light_intensity = self.light_intensity;
    }

    pub fn record_draws(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        frame_index: usize,
    ) {
        let mut current_offset = 0;
        
        for batch in &self.batches {
            let count = batch.instances.len() as u32;
            if count == 0 { continue; }

            builder.bind_vertex_buffers(0, (batch.mesh.vertices.clone(),));
            
            let requested_tex = batch.mesh.base_color_texture.unwrap_or(0);
            let available_textures = self.descriptor_sets[frame_index].len();
            let tex_idx = requested_tex.min(available_textures.saturating_sub(1));

            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics, 
                pipeline.layout().clone(), 
                0, 
                self.descriptor_sets[frame_index][tex_idx].clone()
            );

            if let Some(indices) = &batch.mesh.indices {
                builder.bind_index_buffer(indices.clone());
                builder.draw_indexed(
                    batch.mesh.index_count, 
                    count, 
                    0, 
                    0, 
                    current_offset
                ).unwrap();
            } else {
                builder.draw(
                    batch.mesh.vertex_count, 
                    count, 
                    0, 
                    current_offset
                ).unwrap();
            }
            
            current_offset += count;
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
        )
        .unwrap();
        let image = ImmutableImage::from_iter::<u8, _, _, _>(
            memory_allocator.as_ref(),
            pixels_rgba.iter().copied(),
            vulkano::image::ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1,
            },
            vulkano::image::MipmapsCount::Log2,
            Format::R8G8B8A8_SRGB,
            &mut upload_builder,
        )
        .unwrap();
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
        pipeline: &Arc<GraphicsPipeline>,
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
        self.ensure_descriptor_cache(pipeline, self.texture_views.len());
    }

    pub fn set_light(&mut self, position: [f32; 3], color: [f32; 3], intensity: f32) {
        self.light_pos = position;
        self.light_color = color;
        self.light_intensity = intensity;
    }
}

pub fn record_compute_physics(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, 
    compute_pipeline: &Arc<ComputePipeline>,
    compute_set: &Arc<PersistentDescriptorSet>,
    total_objects: u32,
    dt: f32
) {
    let workgroups_x = (total_objects + 255) / 256;
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute, 
            compute_pipeline.layout().clone(), 
            0, 
            compute_set.clone()
        )
        .push_constants(
            compute_pipeline.layout().clone(), 
            0, 
            PhysicsPushConstants { dt, object_count: total_objects }
        )
        .dispatch([workgroups_x, 1, 1])
        .unwrap();

}

pub fn begin_render_pass_only(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, 
    framebuffers: &[Arc<Framebuffer>], 
    img_index: u32, 
    dims: [u32; 2], 
    pipeline: &Arc<GraphicsPipeline>
) {
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![
                    Some([0.01, 0.01, 0.02, 1.0].into()), // Dark blue clear
                    Some(1.0.into()),
                ],
                ..RenderPassBeginInfo::framebuffer(framebuffers[img_index as usize].clone())
            },
            SubpassContents::Inline,
        )
        .unwrap() 
        .set_viewport(0, vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dims[0] as f32, dims[1] as f32],
            depth_range: 0.0..1.0,
        }])
        .bind_pipeline_graphics(pipeline.clone());
}