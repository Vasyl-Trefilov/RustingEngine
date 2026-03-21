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
use vulkano::DeviceSize;
use rayon::prelude::*;

use crate::input::MouseState;
use crate::renderer::frustum::{frustum_planes, sphere_visible, view_proj_matrix};
use crate::renderer::pipeline::UniformBufferObject;
use crate::scene::animation::AnimationType;
use crate::scene::object::{Instance, InstanceData, RenderBatch, Texture};
use crate::shapes::Mesh;

pub struct RenderScene {
    pub max_instances: usize,
    pub batches: Vec<RenderBatch>,
    pub frames: Vec<FrameData>,
    pub light_pos: [f32; 3],
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub texture_views: Vec<Arc<ImageView<ImmutableImage>>>,
    pub texture_sampler: Arc<Sampler>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    descriptor_sets: Vec<Vec<Arc<PersistentDescriptorSet>>>,
}

pub struct FrameData {
    pub instance_staging: Subbuffer<[InstanceData]>,
    pub instance_device: Subbuffer<[InstanceData]>,
    pub uniform_buffer: Subbuffer<UniformBufferObject>,
    pub instance_scratch: Vec<InstanceData>,
    pub visible_instance_count: u32,
    pub batch_visible_counts: Vec<u32>, // * Visible instance count per batch index (matches `batches` order).
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

        for _ in 0..frames_in_flight {
            let instance_staging = Buffer::new_slice::<InstanceData>(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                max_instances as u64,
            )
            .unwrap();

            let instance_device = Buffer::new_slice::<InstanceData>(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::DeviceOnly,
                    ..Default::default()
                },
                max_instances as u64,
            )
            .unwrap();

            let uniform_buffer = Buffer::from_data(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                UniformBufferObject::default(),
            )
            .unwrap();

            let mut instance_scratch = Vec::new();
            instance_scratch.reserve_exact(max_instances);

            frames.push(FrameData {
                instance_staging,
                instance_device,
                uniform_buffer,
                instance_scratch,
                visible_instance_count: 0,
                batch_visible_counts: Vec::new(),
            });
        }

        let mut scene = Self {
            max_instances,
            batches: Vec::new(),
            frames,
            light_pos: [0.0, 10.0, 0.0],
            light_color: [1.0, 1.0, 1.0],
            light_intensity: 50.0,
            texture_views: vec![default_texture_view],
            texture_sampler,
            descriptor_set_allocator: descriptor_set_allocator.clone(),
            descriptor_sets: vec![Vec::new(); frames_in_flight],
        };

        scene.ensure_descriptor_cache(pipeline, scene.texture_views.len());
        scene
    }

    fn ensure_descriptor_cache(&mut self, pipeline: &Arc<GraphicsPipeline>, target_texture_count: usize) {
        let layout = pipeline.layout().set_layouts()[0].clone();
        for (frame_i, frame) in self.frames.iter().enumerate() {
            while self.descriptor_sets[frame_i].len() < target_texture_count {
                let tex_i = self.descriptor_sets[frame_i].len();
                let set = PersistentDescriptorSet::new(
                    self.descriptor_set_allocator.as_ref(),
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, frame.uniform_buffer.clone()),
                        WriteDescriptorSet::image_view_sampler(
                            1,
                            self.texture_views[tex_i].clone(),
                            self.texture_sampler.clone(),
                        ),
                        WriteDescriptorSet::buffer(2, frame.instance_device.clone()),
                    ],
                )
                .unwrap();
                self.descriptor_sets[frame_i].push(set);
            }
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

    pub fn update(&mut self, elapsed: f32, _mouse: &MouseState) {
        for batch in &mut self.batches {
            batch.instances.par_iter_mut().for_each(|inst| {
                match &inst.animation {
                    AnimationType::Custom(logic) => {
                        logic(
                            &mut inst.transform,
                            &mut inst.velocity,
                            &mut inst.original_position,
                            &mut inst.color,
                            elapsed,
                        );
                    }
                    AnimationType::Rotate => {
                        inst.transform.rotation[0] = elapsed * 0.2;
                        inst.transform.rotation[1] = elapsed * 0.2;
                    }
                    _ => {}
                }

                inst.model_matrix = inst.transform.to_matrix();
            });
        }
    }

    // * CPU work: UBO, frustum culling, parallel instance packing, staging buffer write.
    pub fn prepare_frame(
        &mut self,
        frame_index: usize,
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        eye_pos: [f32; 3],
    ) {
        let max_instances = self.max_instances;
        let frame = &mut self.frames[frame_index];
        let total_instances: usize = self.batches.iter().map(|b| b.instances.len()).sum();
        if total_instances == 0 {
            frame.visible_instance_count = 0;
            frame.batch_visible_counts.clear();
            return;
        }

        {
            let mut ubo_data = frame.uniform_buffer.write().unwrap();
            ubo_data.view = view;
            ubo_data.proj = proj;
            ubo_data.eye_pos = eye_pos;
            ubo_data.light_pos = self.light_pos;
            ubo_data.light_color = self.light_color;
            ubo_data.light_intensity = self.light_intensity;
        }

        let vp = view_proj_matrix(&view, &proj);
        let planes = frustum_planes(&vp);

        frame.instance_scratch.clear();
        frame.batch_visible_counts.resize(self.batches.len(), 0);

        for (bi, batch) in self.batches.iter().enumerate() {
            if batch.instances.is_empty() {
                continue;
            }
            let chunk: Vec<InstanceData> = batch
                .instances
                .par_iter()
                .filter(|inst| {
                    let s = inst.transform.scale;
                    let r = s[0].max(s[1]).max(s[2]) * 1.5;
                    sphere_visible(&planes, inst.transform.position, r)
                })
                .map(|inst| InstanceData {
                    model: inst.model_matrix,
                    color: inst.color,
                    emissive: inst.emissive,
                    mat_props: [inst.roughness, inst.metalness, 0.0, 0.0],
                })
                .collect();
            frame.batch_visible_counts[bi] = chunk.len() as u32;
            frame.instance_scratch.extend(chunk);
        }

        let visible = frame.instance_scratch.len();
        assert!(visible <= max_instances);
        frame.visible_instance_count = visible as u32;

        if visible == 0 {
            return;
        }

        let mut staging = frame.instance_staging.write().unwrap();
        staging[..visible].copy_from_slice(&frame.instance_scratch[..visible]);
    }

    // * GPU transfer: staging → device instance buffer + barriers.
    pub fn record_instance_transfer(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        frame_index: usize,
    ) {
        let frame = &self.frames[frame_index];
        let visible = frame.visible_instance_count as usize;
        if visible == 0 {
            return;
        }

        let mut copy_info = CopyBufferInfoTyped::buffers(
            frame.instance_staging.clone(),
            frame.instance_device.clone(),
        );
        copy_info.regions[0].size = visible as DeviceSize;
        builder.copy_buffer(CopyBufferInfo::from(copy_info)).unwrap();
    }

    pub fn record_draws(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        frame_index: usize,
    ) {
        let visible_total = self.frames[frame_index].visible_instance_count;
        if visible_total == 0 {
            return;
        }

        let counts = self.frames[frame_index].batch_visible_counts.clone();

        let mut current_instance_offset: u32 = 0;
        for (bi, batch) in self.batches.iter().enumerate() {
            if batch.instances.is_empty() {
                continue;
            }

            let visible_in_batch = *counts.get(bi).unwrap_or(&0);
            if visible_in_batch == 0 {
                continue;
            }

            builder.bind_vertex_buffers(0, (batch.mesh.vertices.clone(),));

            let texture_index = batch
                .mesh
                .base_color_texture
                .filter(|idx| *idx < self.texture_views.len())
                .unwrap_or(0);
            let descriptor_set = self.descriptor_sets[frame_index][texture_index].clone();

            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            );

            if let Some(indices) = &batch.mesh.indices {
                builder.bind_index_buffer(indices.clone());
                builder
                    .draw_indexed(
                        batch.mesh.index_count,
                        visible_in_batch,
                        0,
                        0,
                        current_instance_offset,
                    )
                    .unwrap();
            } else {
                builder
                    .draw(
                        batch.mesh.vertex_count,
                        visible_in_batch,
                        0,
                        current_instance_offset,
                    )
                    .unwrap();
            }

            current_instance_offset += visible_in_batch;
        }
    }

    pub fn set_light(&mut self, position: [f32; 3], color: [f32; 3], intensity: f32) {
        self.light_pos = position;
        self.light_color = color;
        self.light_intensity = intensity;
    }
}
