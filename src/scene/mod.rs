pub mod animation;
pub mod object;

use crate::scene::object::PhysicsPushConstants;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyBufferInfoTyped,
    PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::ImmutableImage;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Framebuffer;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::sync::GpuFuture;
use vulkano::DeviceSize;

use crate::geometry::Mesh;
use crate::input::MouseState;
use crate::rendering::compute_registry::ComputeShaderRegistry;
use crate::rendering::compute_registry::ComputeShaderType;
use crate::rendering::pipeline::UniformBufferObject;
use crate::rendering::shader_registry::{ShaderRegistry, ShaderType};
use crate::scene::animation::AnimationType;
use crate::scene::object::{Instance, InstanceData, RenderBatch, Texture, Transform};
use vulkano::command_buffer::{DrawIndexedIndirectCommand, DrawIndirectCommand};

#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MeshPushConstants {
    pub visible_list_offset: u32,
    pub use_culling: u32, // 0 = direct, 1 = indirect
}

pub struct ComputeDispatchInfo {
    pub compute_shader: ComputeShaderType,
    pub offset: u32,
    pub count: u32,
}

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
    pub big_objects_indices: Subbuffer<[u32]>,
    pub num_big_objects: u32,
    pub grid_counts: Subbuffer<[u32]>,
    pub grid_objects: Subbuffer<[u32]>,
    pub visible_indices: Subbuffer<[u32]>,
    pub total_instances: u32,
    pub max_object_radius: f32,
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
    pub fn add_instance(
        &mut self,
        mesh: Mesh,
        instance: Instance,
        allocator: &Arc<StandardMemoryAllocator>,
    ) -> InstanceHandle {
        let shader = instance.shader;
        let compute_shader = instance.physics.compute_shader;
        for (batch_index, batch) in self.batches.iter_mut().enumerate() {
            if batch.mesh.vertices.buffer() == mesh.vertices.buffer()
                && batch.shader == shader
                && batch.compute_shader == compute_shader
            {
                batch.instances.push(instance);

                return InstanceHandle {
                    batch_index,
                    instance_index: batch.instances.len() - 1,
                };
            }
        }

        self.batches.push(RenderBatch {
            mesh,
            shader,
            compute_shader,
            instances: vec![instance],
            base_instance_offset: 0,
            indirect_buffer: Buffer::new_slice::<DrawIndexedIndirectCommand>(
                allocator,
                BufferCreateInfo {
                    usage: BufferUsage::INDIRECT_BUFFER
                        | BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::DeviceOnly,
                    ..Default::default()
                },
                1,
            )
            .unwrap(),
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
        let default_texture =
            Self::create_texture_image(memory_allocator, queue, &[255, 255, 255, 255], 1, 1);
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
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            max_instances as u64,
        )
        .unwrap();
        let physics_write = Buffer::new_slice::<InstanceData>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            max_instances as u64,
        )
        .unwrap();

        for _ in 0..frames_in_flight {
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

            frames.push(FrameData { uniform_buffer });
        }
        let hash_size = 65521;
        let max_per_cell = 128;

        let big_objects_indices = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            1024,
        )
        .unwrap();

        let grid_counts = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            hash_size,
        )
        .unwrap();

        let grid_objects = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            hash_size * max_per_cell,
        )
        .unwrap();

        let visible_indices = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            1_000_000,
        )
        .unwrap();

        Self {
            batches: Vec::new(),
            frames,
            light_pos: [0.0, 10.0, 0.0],
            light_color: [1.0, 1.0, 1.0],
            light_intensity: 50.0,
            texture_views: vec![default_texture_view],
            texture_sampler,
            descriptor_set_allocator: descriptor_set_allocator.clone(),
            descriptor_sets: vec![Vec::new(); frames_in_flight],
            total_instances: 0,
            physics_read,
            physics_write,
            big_objects_indices: big_objects_indices,
            num_big_objects: 1024,
            grid_counts,
            grid_objects,
            visible_indices,
            max_object_radius: 0.0,
        }
    }

    pub fn upload_to_gpu(
        &mut self,
        allocator: &Arc<StandardMemoryAllocator>,
        queue: &Arc<Queue>,
        compute_registry: &ComputeShaderRegistry,
    ) -> Vec<ComputeDispatchInfo> {
        let total_instances = self
            .batches
            .iter()
            .map(|batch| batch.instances.len())
            .sum::<usize>();
        self.total_instances = total_instances as u32;

        if total_instances == 0 {
            return vec![];
        }

        let override_shader = compute_registry.scene_shader_optional();
        if override_shader.is_none() {
            self.batches
                .sort_by_key(|b| (b.compute_shader.sort_key(), b.shader.sort_key()));
        }

        let mut flat_data = Vec::with_capacity(total_instances);
        let mut big_indices: Vec<u32> = Vec::new();
        let mut max_small_radius = 0.1;
        let threshold = 2.5;
        let mut current_idx = 0;

        for batch in &self.batches {
            for inst in &batch.instances {
                let m = inst.model_matrix;
                let scale = f32::max(
                    f32::max(
                        (m[0][0].powi(2) + m[0][1].powi(2) + m[0][2].powi(2)).sqrt(),
                        (m[1][0].powi(2) + m[1][1].powi(2) + m[1][2].powi(2)).sqrt(),
                    ),
                    (m[2][0].powi(2) + m[2][1].powi(2) + m[2][2].powi(2)).sqrt(),
                );

                let radius = scale * 0.5;

                if radius > threshold {
                    big_indices.push(current_idx as u32);
                } else {
                    if radius > max_small_radius {
                        max_small_radius = radius;
                    }
                }

                flat_data.push(InstanceData {
                    model: inst.model_matrix,
                    color: [inst.color[0], inst.color[1], inst.color[2], inst.emissive],
                    mat_props: [inst.roughness, inst.metalness, 0.0, 0.0],
                    velocity: [
                        inst.physics.linear_velocity[0],
                        inst.physics.linear_velocity[1],
                        inst.physics.linear_velocity[2],
                        inst.physics.bounciness,
                    ],
                    angular_velocity: [
                        inst.physics.angular_velocity[0],
                        inst.physics.angular_velocity[1],
                        inst.physics.angular_velocity[2],
                        inst.physics.friction,
                    ],
                    physic_props: [
                        inst.physics.collision_type.sort_key(),
                        inst.physics.mass,
                        inst.physics.gravity_scale,
                        0.0,
                    ],
                });
                current_idx += 1;
            }
        }

        self.max_object_radius = max_small_radius;
        self.num_big_objects = big_indices.len() as u32;

        let staging = Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            flat_data,
        )
        .unwrap();

        let cmd_allocator =
            StandardCommandBufferAllocator::new(queue.device().clone(), Default::default());
        let mut builder = AutoCommandBufferBuilder::primary(
            &cmd_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let copy_count = self.total_instances as u64;
        builder
            .copy_buffer(CopyBufferInfoTyped::buffers(
                staging.clone(),
                self.physics_read.clone().slice(0..copy_count),
            ))
            .unwrap();
        builder
            .copy_buffer(CopyBufferInfoTyped::buffers(
                staging.clone(),
                self.physics_write.clone().slice(0..copy_count),
            ))
            .unwrap();

        let mut indirect_data: Vec<DrawIndexedIndirectCommand> = Vec::new();
        for batch in &self.batches {
            indirect_data.push(DrawIndexedIndirectCommand {
                index_count: batch.mesh.index_count,
                instance_count: batch.instances.len() as u32,
                first_index: 0,
                vertex_offset: 0,
                first_instance: 0,
            });
        }

        let indirect_staging = Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            indirect_data,
        )
        .unwrap();

        let mut indirect_offset = 0u64;
        let cmd_size = std::mem::size_of::<DrawIndexedIndirectCommand>() as u64;

        for batch in &self.batches {
            let cmd = DrawIndexedIndirectCommand {
                index_count: batch.mesh.index_count,
                instance_count: batch.instances.len() as u32,
                first_index: 0,
                vertex_offset: 0,
                first_instance: 0,
            };
            let single_staging = Buffer::from_iter(
                allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                std::iter::once(cmd),
            )
            .unwrap();
            builder
                .copy_buffer(CopyBufferInfoTyped::buffers(
                    single_staging,
                    batch.indirect_buffer.clone(),
                ))
                .unwrap();
            indirect_offset += cmd_size;
        }

        if !big_indices.is_empty() {
            let big_staging = Buffer::from_iter(
                allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                big_indices,
            )
            .unwrap();
            builder
                .copy_buffer(CopyBufferInfoTyped::buffers(
                    big_staging,
                    self.big_objects_indices
                        .clone()
                        .slice(0..self.num_big_objects as u64),
                ))
                .unwrap();
        }

        builder
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        if let Some(shader_type) = override_shader {
            vec![ComputeDispatchInfo {
                compute_shader: shader_type,
                offset: 0,
                count: self.total_instances,
            }]
        } else {
            let mut dispatches = Vec::new();
            let mut current_offset = 0;
            for batch in &mut self.batches {
                let count = batch.instances.len() as u32;
                batch.base_instance_offset = current_offset;
                dispatches.push(ComputeDispatchInfo {
                    compute_shader: batch.compute_shader,
                    offset: current_offset,
                    count,
                });
                current_offset += count;
            }
            dispatches
        }
    }

    pub fn ensure_descriptor_cache(
        &mut self,
        pipeline: &Arc<GraphicsPipeline>,
        target_tex_count: usize,
    ) {
        let layout = pipeline.layout().set_layouts()[0].clone();

        for (frame_i, frame) in self.frames.iter().enumerate() {
            let total_sets_needed = target_tex_count * 2;

            if self.descriptor_sets[frame_i].len() == total_sets_needed {
                continue;
            }

            self.descriptor_sets[frame_i].clear();

            for tex_idx in 0..target_tex_count {
                // Buffer read
                let set_a = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, frame.uniform_buffer.clone()),
                        WriteDescriptorSet::image_view_sampler(
                            1,
                            self.texture_views[tex_idx].clone(),
                            self.texture_sampler.clone(),
                        ),
                        WriteDescriptorSet::buffer(2, self.physics_read.clone()),
                        WriteDescriptorSet::buffer(3, self.visible_indices.clone()),
                    ],
                )
                .unwrap();

                // Buffer write
                let set_b = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, frame.uniform_buffer.clone()),
                        WriteDescriptorSet::image_view_sampler(
                            1,
                            self.texture_views[tex_idx].clone(),
                            self.texture_sampler.clone(),
                        ),
                        WriteDescriptorSet::buffer(2, self.physics_write.clone()),
                        WriteDescriptorSet::buffer(3, self.visible_indices.clone()),
                    ],
                )
                .unwrap();

                self.descriptor_sets[frame_i].push(set_a);
                self.descriptor_sets[frame_i].push(set_b);
            }
        }
    }

    /// Draw all batches using a single pipeline (backward compatible).
    /// Used by main.rs which creates its own pipeline directly.
    pub fn record_draws(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        frame_index: usize,
        physics_buffer_index: usize,
    ) {
        let mut current_offset = 0;

        builder.bind_pipeline_graphics(pipeline.clone());

        for batch in &self.batches {
            let count = batch.instances.len() as u32;
            if count == 0 {
                continue;
            }

            builder.bind_vertex_buffers(0, (batch.mesh.vertices.clone(),));

            let requested_tex = batch.mesh.base_color_texture.unwrap_or(0);
            let descriptor_idx = (requested_tex * 2) + physics_buffer_index;

            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                self.descriptor_sets[frame_index][descriptor_idx].clone(),
            );

            if let Some(indices) = &batch.mesh.indices {
                builder.bind_index_buffer(indices.clone());
                builder
                    .draw_indexed(batch.mesh.index_count, count, 0, 0, current_offset)
                    .unwrap();
            } else {
                builder
                    .draw(batch.mesh.vertex_count, count, 0, current_offset)
                    .unwrap();
            }

            current_offset += count;
        }
    }

    /// Draw all batches using the multi-shader registry.
    /// Sorts batches by shader type to minimize pipeline switches.
    pub fn record_draws_multi(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        registry: &ShaderRegistry,
        frame_index: usize,
        physics_buffer_index: usize,
        use_culling: bool,
    ) {
        let mut last_shader: Option<ShaderType> = None;

        for batch in &self.batches {
            if batch.instances.is_empty() {
                continue;
            }

            let effective_shader = registry.resolve_shader(batch.shader);
            let pipeline = registry.get_pipeline(effective_shader);

            if last_shader != Some(effective_shader) {
                builder.bind_pipeline_graphics(pipeline.clone());
                last_shader = Some(effective_shader);
            }

            builder.push_constants(
                pipeline.layout().clone(),
                0,
                MeshPushConstants {
                    visible_list_offset: batch.base_instance_offset,
                    use_culling: if use_culling { 1 } else { 0 },
                },
            );

            builder.bind_vertex_buffers(0, (batch.mesh.vertices.clone(),));

            let requested_tex = batch.mesh.base_color_texture.unwrap_or(0);
            let descriptor_idx = (requested_tex * 2) + physics_buffer_index;

            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                self.descriptor_sets[frame_index][descriptor_idx].clone(),
            );

            if let Some(indices) = &batch.mesh.indices {
                builder.bind_index_buffer(indices.clone());
            
                builder
                    .draw_indexed_indirect(batch.indirect_buffer.clone())
                    .unwrap();
            } else {
                builder
                    .draw(
                        batch.mesh.vertex_count,
                        batch.instances.len() as u32,
                        0,
                        batch.base_instance_offset,
                    )
                    .unwrap();
            }
        }
    }
    pub fn prepare_frame_ubo(
        &mut self,
        frame_index: usize,
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        eye_pos: [f32; 3],
    ) {
        let mut ubo = self.frames[frame_index].uniform_buffer.write().unwrap();
        ubo.view = view;
        ubo.proj = proj;
        ubo.eye_pos = eye_pos;
        ubo.light_pos = self.light_pos;
        ubo.light_color = self.light_color;
        ubo.light_intensity = self.light_intensity;
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
            let image = Self::create_texture_image(
                memory_allocator,
                queue,
                &pixels_rgba,
                tex.width,
                tex.height,
            );
            let view = ImageView::new_default(image).unwrap();
            self.texture_views.push(view);
        }
        self.ensure_descriptor_cache(pipeline, textures.len());
    }

    pub fn set_light(&mut self, position: [f32; 3], color: [f32; 3], intensity: f32) {
        self.light_pos = position;
        self.light_color = color;
        self.light_intensity = intensity;
    }
}

pub fn record_compute_physics(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    compute_pipeline: &Arc<vulkano::pipeline::ComputePipeline>,
    compute_set: &Arc<PersistentDescriptorSet>,
    max_instances: u32,
    dt: f32,
    total_objects: u32,
    num_big_objects: u32,
) {
    let workgroups_x = (max_instances as u32 + 255) / 256;
    if workgroups_x == 0 {
        return;
    }
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            compute_set.clone(),
        )
        .push_constants(
            compute_pipeline.layout().clone(),
            0,
            crate::scene::object::PhysicsPushConstants {
                dt,
                total_objects,
                offset: 0,
                count: max_instances as u32,
                num_big_objects: num_big_objects,
                _pad: [0, 0, 0],
                global_gravity: [0.0, -9.81, 0.0, 2.0],
            },
        )
        .dispatch([workgroups_x, 1, 1])
        .unwrap();
}

pub fn record_compute_physics_multi(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    registry: &crate::rendering::compute_registry::ComputeShaderRegistry,
    compute_sets: &HashMap<
        ComputeShaderType,
        (Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>),
    >,
    grid_build_sets: &(Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>),
    grid_counts: &Subbuffer<[u32]>,
    dispatches: &[ComputeDispatchInfo],
    dt: f32,
    total_objects: u32,
    cell_size: f32,
    num_big_objects: u32,
    ping_pong: bool,
) {
    builder.fill_buffer(grid_counts.clone(), 0u32).unwrap();
    let build_pipeline = registry.get_pipeline(ComputeShaderType::GridBuild);
    let grid_set = if ping_pong {
        &grid_build_sets.1
    } else {
        &grid_build_sets.0
    };
    builder
        .bind_pipeline_compute(build_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            build_pipeline.layout().clone(),
            0,
            grid_set.clone(),
        )
        .push_constants(
            build_pipeline.layout().clone(),
            0,
            PhysicsPushConstants {
                dt,
                total_objects,
                offset: 0,
                count: total_objects,
                num_big_objects: num_big_objects,
                _pad: [0, 0, 0],
                global_gravity: [0.0, -9.81, 0.0, cell_size], // w = CELL_SIZE (must be >= 2 * max_object_radius)
            },
        )
        .dispatch([(total_objects + 255) / 256, 1, 1])
        .unwrap();

    let mut last_bound = None;

    for dispatch in dispatches {
        let shader_to_use = dispatch.compute_shader;
        let compute_pipeline = registry.get_pipeline(shader_to_use);

        if shader_to_use == ComputeShaderType::GridBuild {
            continue;
        } else {
            let (set_0, set_1) = compute_sets.get(&shader_to_use).unwrap();
            let compute_set = if ping_pong { set_1 } else { set_0 };
            if last_bound != Some(shader_to_use) {
                builder.bind_pipeline_compute(compute_pipeline.clone());
                builder.bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0,
                    compute_set.clone(),
                );
                last_bound = Some(shader_to_use);
            }
            let workgroups_x = (dispatch.count + 255) / 256;
            if workgroups_x > 0 {
                builder
                    .push_constants(
                        compute_pipeline.layout().clone(),
                        0,
                        PhysicsPushConstants {
                            dt,
                            total_objects,
                            offset: dispatch.offset,
                            count: dispatch.count,
                            num_big_objects: num_big_objects,
                            _pad: [0, 0, 0],
                            global_gravity: [0.0, -9.81, 0.0, cell_size],
                        },
                    )
                    .dispatch([workgroups_x, 1, 1])
                    .unwrap();
            }
        }
    }
}
pub fn begin_render_pass_only(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    framebuffers: &[Arc<Framebuffer>],
    img_index: u32,
    dims: [u32; 2],
    pipeline: &Arc<GraphicsPipeline>,
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
        .set_viewport(
            0,
            vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dims[0] as f32, dims[1] as f32],
                depth_range: 0.0..1.0,
            }],
        )
        .bind_pipeline_graphics(pipeline.clone());
}

pub fn record_compute_physics_spatial(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    registry: &ComputeShaderRegistry,
    compute_sets: &HashMap<
        ComputeShaderType,
        (Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>),
    >,
    grid_build_sets: &(Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>),
    grid_counts: &Subbuffer<[u32]>,
    dispatches: &[ComputeDispatchInfo],
    dt: f32,
    total_objects: u32,
    cell_size: f32,
    num_big_objects: u32,
    ping_pong: bool,
) {
    let build_pipeline = registry.get_pipeline(ComputeShaderType::GridBuild);

    let mut read_index: usize = 0;

    for dispatch in dispatches {
        builder.fill_buffer(grid_counts.clone(), 0u32).unwrap();

        let grid_set = if read_index == 0 {
            &grid_build_sets.0
        } else {
            &grid_build_sets.1
        };

        builder
            .bind_pipeline_compute(build_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                build_pipeline.layout().clone(),
                0,
                grid_set.clone(),
            )
            .push_constants(
                build_pipeline.layout().clone(),
                0,
                PhysicsPushConstants {
                    dt,
                    total_objects,
                    offset: 0,
                    count: total_objects,
                    num_big_objects: num_big_objects,
                    _pad: [0, 0, 0],
                    global_gravity: [0.0, -9.81, 0.0, cell_size],
                },
            )
            .dispatch([(total_objects + 255) / 256, 1, 1])
            .unwrap();

        let compute_pipeline = registry.get_pipeline(dispatch.compute_shader);
        let (set_0, set_1) = compute_sets.get(&dispatch.compute_shader).unwrap();
        let compute_set = if ping_pong { set_1 } else { set_0 };

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                compute_set.clone(),
            )
            .push_constants(
                compute_pipeline.layout().clone(),
                0,
                PhysicsPushConstants {
                    dt,
                    total_objects,
                    offset: dispatch.offset,
                    count: dispatch.count,
                    num_big_objects: num_big_objects,
                    _pad: [0, 0, 0],
                    global_gravity: [0.0, -9.81, 0.0, 2.0],
                },
            )
            .dispatch([(dispatch.count + 255) / 256, 1, 1])
            .unwrap();

        read_index ^= 1;
    }
}
