use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use vulkano::sync::GpuFuture;
use winit::event::{Event, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::CursorGrabMode;

use crate::core::{Material, Physics, Transform};
use crate::geometry::shapes::{create_cube, create_sphere_subdivided};
use crate::rendering::camera::create_projection_matrix;
use crate::rendering::compute_registry::{ComputeShaderRegistry, ComputeShaderType};
use crate::rendering::init_vulkan;
use crate::rendering::pipeline::create_pipeline;
use crate::rendering::render::create_builder;
use crate::rendering::shader_registry::{ShaderRegistry, ShaderType};
use crate::rendering::swapchain::{create_framebuffers, create_render_pass};
use crate::rendering::VulkanBase;
use crate::scene::object::Instance;
use crate::scene::{
    begin_render_pass_only, record_compute_physics_multi, ComputeDispatchInfo, RenderScene,
};

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::ImageUsage;
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::swapchain::{
    AcquireError, CompositeAlpha, PresentMode, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError};

pub struct PerspectiveCamera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl PerspectiveCamera {
    pub fn new(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            position: [0.0, 50.0, 200.0],
            yaw: 90.0f32.to_radians(),
            pitch: 0.0,
            fov: fov.to_radians(),
            aspect,
            near,
            far,
        }
    }

    pub fn update(
        &mut self,
        keys: &HashSet<VirtualKeyCode>,
        sprint: f32,
        dt: f32,
        mouse_captured: bool,
    ) -> [[f32; 4]; 4] {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();

        let forward = [-self.yaw.cos(), 0.0, -self.yaw.sin()];
        let len = (forward[0] * forward[0] + forward[2] * forward[2]).sqrt();
        let forward = [forward[0] / len, 0.0, forward[2] / len];

        let right = [-self.yaw.sin(), 0.0, self.yaw.cos()];
        let view_forward = [yaw_cos * pitch_cos, pitch_sin, yaw_sin * pitch_cos];

        if mouse_captured {
            let speed = 0.1 * sprint;
            if keys.contains(&VirtualKeyCode::W) {
                for i in 0..3 {
                    self.position[i] += forward[i] * speed;
                }
            }
            if keys.contains(&VirtualKeyCode::S) {
                for i in 0..3 {
                    self.position[i] -= forward[i] * speed;
                }
            }
            if keys.contains(&VirtualKeyCode::A) {
                for i in 0..3 {
                    self.position[i] -= right[i] * speed;
                }
            }
            if keys.contains(&VirtualKeyCode::D) {
                for i in 0..3 {
                    self.position[i] += right[i] * speed;
                }
            }
            if keys.contains(&VirtualKeyCode::Space) {
                self.position[1] += speed;
            }
            if keys.contains(&VirtualKeyCode::LControl) {
                self.position[1] -= speed;
            }
        }

        let target = [
            self.position[0] + view_forward[0],
            self.position[1] + view_forward[1],
            self.position[2] + view_forward[2],
        ];

        crate::rendering::camera::create_look_at(self.position, target, [0.0, 1.0, 0.0])
    }
}

/// High-level engine structure for building and rendering scenes.
///
/// Handles Vulkan context, swapchain, render pass, rendering pipeline, inputs,
/// and the physics simulation loop.
pub struct Engine {
    pub camera: Arc<Mutex<PerspectiveCamera>>,
    scene: Arc<Mutex<RenderScene>>,
    event_loop: Option<EventLoop<()>>,
    base: VulkanBase,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<vulkano::image::SwapchainImage>>,
    render_pass: Arc<vulkano::render_pass::RenderPass>,
    registry: ShaderRegistry,
    compute_registry: ComputeShaderRegistry,
    memory_allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    cached_cube_mesh: Option<crate::geometry::Mesh>,
    cached_sphere_mesh: Option<crate::geometry::Mesh>,
}

impl Engine {
    /// Creates a new Engine and initializes Vulkan, Winit, and rendering pipelines.
    ///
    /// # Arguments
    /// * `title` - The window title.
    ///
    /// # Examples
    /// ```
    /// let engine = rusting_engine::Engine::new("My Game");
    /// ```
    pub fn new(title: &str) -> Self {
        let event_loop = EventLoop::new();
        let base = init_vulkan(&event_loop, title);
        let dims = base.window.inner_size();

        let (swapchain, images) = Swapchain::new(
            base.device.clone(),
            base.surface.clone(),
            SwapchainCreateInfo {
                min_image_count: 3,
                image_format: None,
                image_extent: [dims.width, dims.height],
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: CompositeAlpha::Opaque,
                present_mode: PresentMode::Immediate, // FASTEST MODE
                ..Default::default()
            },
        )
        .unwrap();

        let render_pass = create_render_pass(base.device.clone(), &swapchain);
        let registry = ShaderRegistry::new(&base.device, &render_pass);

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            base.device.clone(),
            Default::default(),
        ));
        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(base.device.clone()));
        let mem_allocator = Arc::new(
            vulkano::memory::allocator::StandardMemoryAllocator::new_default(base.device.clone()),
        );

        let scene = RenderScene::new(
            &mem_allocator,
            &ds_allocator,
            registry.default_pipeline(),
            &base.queue,
            3,
            1_000_000, // 1m
        );

        let compute_registry = ComputeShaderRegistry::new(&base.device);

        Self {
            event_loop: Some(event_loop),
            base,
            swapchain,
            images,
            render_pass,
            registry,
            compute_registry,
            memory_allocator: mem_allocator,
            descriptor_set_allocator: ds_allocator,
            command_buffer_allocator: cb_allocator,
            scene: Arc::new(Mutex::new(scene)),
            camera: Arc::new(Mutex::new(PerspectiveCamera::new(
                45.0,
                dims.width as f32 / dims.height as f32,
                0.1,
                1000.0,
            ))),
            cached_cube_mesh: None,
            cached_sphere_mesh: None,
        }
    }

    /// Set the sun in scene
    pub fn set_light(&mut self, pos: [f32; 3], color: [f32; 3], intensity: f32) {
        self.scene.lock().unwrap().set_light(pos, color, intensity);
    }

    /// Adding a cube to the scene.
    ///
    /// # Example
    /// ```
    /// engine.add_cube(
    ///    Transform {
    ///        ..Default::default()
    ///    },
    ///    &Material::standard()
    ///        .build(),
    ///    &Physics::default()
    ///        .collision(0.2), // type, if collision < 0.5 => Box, collision > 0.5 => Sphere
    ///    );
    /// ```
    pub fn add_cube(&mut self, transform: Transform, mat: &Material, phys: &Physics) {
        // Cache mesh on first use, then reuse
        let mesh = if let Some(cached) = &self.cached_cube_mesh {
            cached.clone()
        } else {
            let m = crate::geometry::shapes::create_cube(&self.memory_allocator);
            self.cached_cube_mesh = Some(m.clone());
            m
        };

        let inst = Instance {
            model_matrix: transform.to_matrix(),
            color: mat.color,
            physics: *phys,
            shader: mat.shader,
            ..Default::default()
        };
        self.scene.lock().unwrap().add_instance(mesh, inst);
    }

    /// Adding a sphere to the scene.
    ///
    /// # Example
    /// ```
    /// engine.add_sphere(
    ///    Transform {
    ///        ..Default::default()
    ///    },
    ///    &Material::standard()
    ///        .build(),
    ///    &Physics::default()
    ///        .collision(0.8), // type, if collision < 0.5 => Box, collision > 0.5 => Sphere
    ///    );
    /// ```
    pub fn add_sphere(
        &mut self,
        transform: Transform,
        mat: &Material,
        phys: &Physics,
        subdiv: u32,
    ) {
        // Cache mesh on first use, then reuse
        let mesh = if let Some(cached) = &self.cached_sphere_mesh {
            cached.clone()
        } else {
            let m =
                crate::geometry::shapes::create_sphere_subdivided(&self.memory_allocator, subdiv);
            self.cached_sphere_mesh = Some(m.clone());
            m
        };

        let inst = Instance {
            model_matrix: transform.to_matrix(),
            color: mat.color,
            physics: *phys,
            shader: mat.shader,
            ..Default::default()
        };
        self.scene.lock().unwrap().add_instance(mesh, inst);
    }

    /// Set a scene-wide shader override. All objects will use this shader.
    pub fn set_scene_shader(&mut self, shader: ShaderType) {
        self.registry.set_scene_shader(shader);
    }

    /// Clear the scene-wide shader override. Objects will use their per-object shader.
    pub fn clear_scene_shader(&mut self) {
        self.registry.clear_scene_shader();
    }

    /// Set a scene-wide shader override. All objects will use this shader.
    pub fn set_scene_physic(&mut self, shader: ComputeShaderType) {
        self.compute_registry.set_scene_shader(shader);
    }

    /// Clear the scene-wide shader override. Objects will use their per-object shader.
    pub fn clear_scene_physic(&mut self) {
        self.compute_registry.clear_scene_shader();
    }

    /// Starts the engine's main loop.
    ///
    /// This method will block the current thread until the window is closed.
    /// It handles event dispatch, physics update intervals, and issues continuous draw calls.
    pub fn run(mut self) {
        eprintln!("[DBG] run() start");
        let mut previous_frame_end: Option<Box<dyn GpuFuture>> =
            Some(sync::now(self.base.device.clone()).boxed());
        eprintln!("[DBG] starting upload");
        let (physics_read, physics_write, mut solid_obj_count, dispatches) = {
            let mut s = self.scene.lock().unwrap();
            let d = s.upload_to_gpu(
                &self.memory_allocator,
                &self.base.queue,
                &self.compute_registry,
            );
            eprintln!("[DBG] upload done {}", d.len());
            let tex_count = s.texture_views.len();
            s.ensure_descriptor_cache(self.registry.default_pipeline(), tex_count);
            (
                s.physics_read.clone(),
                s.physics_write.clone(),
                s.total_instances,
                d,
            )
        };
        eprintln!("[DBG] creating compute_sets");

        
        let mut compute_sets: HashMap<
            ComputeShaderType,
            (Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>),
        > = HashMap::new();

        let mut used_shaders = HashSet::new();
        for dispatch in &dispatches {
            used_shaders.insert(dispatch.compute_shader);
        }

        if let Some(scene_shader) = self.compute_registry.scene_shader_optional() {
            used_shaders.insert(scene_shader);
        }

        for shader in used_shaders {
            let compute_layout = self
                .compute_registry
                .get_pipeline(shader)
                .layout()
                .set_layouts()[0]
                .clone();

            let bindings = shader.needs_bindings();
            let scene = self.scene.lock().unwrap();

            let mut writes_0: Vec<WriteDescriptorSet> = vec![];
            let mut writes_1: Vec<WriteDescriptorSet> = vec![];

            if bindings.needs_read_buffer {
                writes_0.push(WriteDescriptorSet::buffer(0, physics_read.clone()));
                writes_1.push(WriteDescriptorSet::buffer(0, physics_write.clone()));
            }
            if bindings.needs_write_buffer {
                writes_0.push(WriteDescriptorSet::buffer(1, physics_write.clone()));
                writes_1.push(WriteDescriptorSet::buffer(1, physics_read.clone()));
            }
            if bindings.needs_grid_counts {
                writes_0.push(WriteDescriptorSet::buffer(2, scene.grid_counts.clone()));
                writes_1.push(WriteDescriptorSet::buffer(2, scene.grid_counts.clone()));
            }
            if bindings.needs_grid_objects {
                writes_0.push(WriteDescriptorSet::buffer(3, scene.grid_objects.clone()));
                writes_1.push(WriteDescriptorSet::buffer(3, scene.grid_objects.clone()));
            }
            if bindings.needs_big_indices {
                writes_0.push(WriteDescriptorSet::buffer(
                    4,
                    scene.big_objects_indices.clone(),
                ));
                writes_1.push(WriteDescriptorSet::buffer(
                    4,
                    scene.big_objects_indices.clone(),
                ));
            }

            let set_0 = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                compute_layout.clone(),
                writes_0,
            )
            .unwrap();

            let set_1 = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                compute_layout.clone(),
                writes_1,
            )
            .unwrap();

            compute_sets.insert(shader, (set_0, set_1));
        }
        eprintln!("[DBG] compute_sets done");

        let grid_build_sets = {
            let scene = self.scene.lock().unwrap();
            let grid_layout = self
                .compute_registry
                .get_pipeline(ComputeShaderType::GridBuild)
                .layout()
                .set_layouts()[0]
                .clone();

            let gb_set_0 = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                grid_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, physics_read.clone()),
                    WriteDescriptorSet::buffer(2, scene.grid_counts.clone()),
                    WriteDescriptorSet::buffer(3, scene.grid_objects.clone()),
                ],
            )
            .unwrap();

            let gb_set_1 = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                grid_layout,
                [
                    WriteDescriptorSet::buffer(0, physics_write.clone()),
                    WriteDescriptorSet::buffer(2, scene.grid_counts.clone()),
                    WriteDescriptorSet::buffer(3, scene.grid_objects.clone()),
                ],
            )
            .unwrap();

            (gb_set_0, gb_set_1)
        };
        eprintln!("[DBG] grid_build_sets done, starting event loop");

        let mut framebuffers =
            create_framebuffers(&self.images, &self.render_pass, &self.memory_allocator);
        let mut inputs = InputState::default();
        let mut last_frame_instant = Instant::now();
        let mut accumulator = 0.0;
        let fixed_dt = 1.0 / 60.0;
        let mut compute_ping_pong = false;
        let mut recreate_swapchain = false;
        let mut frame_index = 0;

        let mut event_loop = self.event_loop.take().unwrap();
        event_loop.run_return(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => recreate_swapchain = true,
                Event::DeviceEvent {
                    event: winit::event::DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    if inputs.mouse_captured {
                        let mut cam = self.camera.lock().unwrap();
                        cam.yaw -= delta.0 as f32 * 0.001;
                        cam.pitch += delta.1 as f32 * 0.001;
                        cam.pitch = cam.pitch.clamp(-1.5, 1.5);
                    }
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput { input, .. } => {
                        if let Some(code) = input.virtual_keycode {
                            if input.state == winit::event::ElementState::Pressed {
                                if code == VirtualKeyCode::Escape {
                                    inputs.mouse_captured = !inputs.mouse_captured;
                                    let _ = self.base.window.set_cursor_grab(
                                        if inputs.mouse_captured {
                                            CursorGrabMode::Locked
                                        } else {
                                            CursorGrabMode::None
                                        },
                                    );
                                    let _ =
                                        self.base.window.set_cursor_visible(!inputs.mouse_captured);
                                }
                                inputs.keys.insert(code);
                            } else {
                                inputs.keys.remove(&code);
                            }
                        }
                    }
                    _ => {}
                },

                Event::MainEventsCleared => {
                    frame_index = (frame_index + 1) % 3;

                    // frame_count += 1;
                    // if fps_timer.elapsed().as_secs_f32() >= 2.0 {
                    //     println!(
                    //         "FPS: {:.0}",
                    //         frame_count as f32 / fps_timer.elapsed().as_secs_f32()
                    //     );
                    //     frame_count = 0;
                    //     fps_timer = Instant::now();
                    // }

                    let now = Instant::now();
                    let mut delta_time = now.duration_since(last_frame_instant).as_secs_f32();
                    last_frame_instant = now;
                    if delta_time > 0.05 {
                        delta_time = 0.05;
                    }
                    accumulator += delta_time;

                    if recreate_swapchain {
                        let new_size = self.base.window.inner_size();
                        if new_size.width > 0 && new_size.height > 0 {
                            let (new_sw, new_img) = self
                                .swapchain
                                .recreate(SwapchainCreateInfo {
                                    image_extent: new_size.into(),
                                    ..self.swapchain.create_info()
                                })
                                .unwrap();
                            self.swapchain = new_sw;
                            self.images = new_img;
                            framebuffers = create_framebuffers(
                                &self.images,
                                &self.render_pass,
                                &self.memory_allocator,
                            );
                            self.camera.lock().unwrap().aspect =
                                new_size.width as f32 / new_size.height as f32;
                        }
                        recreate_swapchain = false;
                    }

                    let (img_index, suboptimal, acquire_future) =
                        match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("{e}"),
                        };
                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    let (proj, view, cam_pos) = {
                        let mut cam = self.camera.lock().unwrap();
                        let sprint = if inputs.keys.contains(&VirtualKeyCode::LShift) {
                            2.0
                        } else {
                            1.0
                        };
                        let view =
                            cam.update(&inputs.keys, sprint, delta_time, inputs.mouse_captured);
                        let proj = create_projection_matrix(cam.aspect, cam.fov, cam.near, cam.far);
                        let cam_pos = cam.position;
                        (proj, view, cam_pos)
                    };

                    {
                        let mut s = self.scene.lock().unwrap();
                        s.prepare_frame_ubo(frame_index, view, proj, cam_pos);
                        solid_obj_count = s.total_instances;
                        let tex_count = s.texture_views.len();
                        s.ensure_descriptor_cache(self.registry.default_pipeline(), tex_count);
                    }

                    let mut comp_builder =
                        create_builder(&self.command_buffer_allocator, &self.base.queue);
                    let mut physics_ran = false;

                    while accumulator >= fixed_dt {
                        let scene = self.scene.lock().unwrap();
                        let cell_size = scene.max_object_radius * 2.0 + 0.2;
                        record_compute_physics_multi(
                            &mut comp_builder,
                            &self.compute_registry,
                            &compute_sets,
                            &grid_build_sets,
                            &scene.grid_counts,
                            &dispatches,
                            fixed_dt,
                            solid_obj_count,
                            cell_size,
                            scene.num_big_objects,
                            compute_ping_pong,
                        );

                        compute_ping_pong = !compute_ping_pong;
                        accumulator -= fixed_dt;
                        physics_ran = true;
                    }

                    let physics_idx = if compute_ping_pong { 1 } else { 0 };

                    if physics_ran {
                        let comp_cb = comp_builder.build().unwrap();
                        let comp_future = sync::now(self.base.device.clone())
                            .then_execute(self.base.queue.clone(), comp_cb)
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap();

                        comp_future.wait(None).unwrap();
                    }

                    let mut render_builder =
                        create_builder(&self.command_buffer_allocator, &self.base.queue);
                    {
                        let mut s = self.scene.lock().unwrap();
                        begin_render_pass_only(
                            &mut render_builder,
                            &framebuffers,
                            img_index,
                            self.base.window.inner_size().into(),
                            self.registry.default_pipeline(),
                        );
                        s.record_draws_multi(
                            &mut render_builder,
                            &self.registry,
                            frame_index,
                            physics_idx,
                        );
                        render_builder.end_render_pass().unwrap();
                    }

                    let render_cb = render_builder.build().unwrap();

                    let future = sync::now(self.base.device.clone())
                        .join(acquire_future)
                        .then_execute(self.base.queue.clone(), render_cb)
                        .unwrap()
                        .then_swapchain_present(
                            self.base.queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                self.swapchain.clone(),
                                img_index,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(_) => {
                            previous_frame_end = Some(sync::now(self.base.device.clone()).boxed());
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(sync::now(self.base.device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("Flush error: {:?}", e);
                            previous_frame_end = Some(sync::now(self.base.device.clone()).boxed());
                        }
                    }
                }
                _ => (),
            }
        });
    }
}

#[derive(Default)]
struct InputState {
    keys: HashSet<VirtualKeyCode>,
    mouse_captured: bool,
}
