use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use winit::event::{Event, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::CursorGrabMode;

use vulkano::sync::GpuFuture;

use crate::core::{Material, Physics, Transform};
use crate::geometry::shapes::{create_cube, create_sphere_subdivided};
use crate::rendering::camera::create_projection_matrix;
use crate::rendering::init_vulkan;
use crate::rendering::pipeline::create_pipeline;
use crate::rendering::render::create_builder;
use crate::rendering::swapchain::{create_framebuffers, create_render_pass};
use crate::rendering::VulkanBase;
use crate::scene::object::Instance;
use crate::scene::RenderScene;
use crate::scene::{begin_render_pass_only, record_compute_physics};
use crate::shaders::{cs, fs, vs};

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
            position: [0.0, 5.0, 20.0],
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

pub struct Engine {
    pub camera: Arc<Mutex<PerspectiveCamera>>,
    scene: Arc<Mutex<RenderScene>>,
    event_loop: Option<EventLoop<()>>,
    base: VulkanBase,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<vulkano::image::SwapchainImage>>,
    render_pass: Arc<vulkano::render_pass::RenderPass>,
    pipeline: Arc<vulkano::pipeline::GraphicsPipeline>,
    compute_pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Engine {
    pub fn new(title: &str) -> Self {
        let event_loop = EventLoop::new();
        let base = init_vulkan(&event_loop, title);
        let dims = base.window.inner_size();

        let vs = vs::load(base.device.clone()).unwrap();
        let fs = fs::load(base.device.clone()).unwrap();

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
        let pipeline = create_pipeline(vs, fs, &render_pass, &base.device);

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
            &pipeline,
            &base.queue,
            3,
            1_000_000,
        );

        let compute_shader = cs::load(base.device.clone()).unwrap();
        let cp = ComputePipeline::new(
            base.device.clone(),
            compute_shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap();

        Self {
            event_loop: Some(event_loop),
            base,
            swapchain,
            images,
            render_pass,
            pipeline,
            compute_pipeline: cp,
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
        }
    }

    pub fn set_light(&mut self, pos: [f32; 3], color: [f32; 3], intensity: f32) {
        self.scene.lock().unwrap().set_light(pos, color, intensity);
    }

    pub fn add_cube(&mut self, pos: [f32; 3], mat: &Material, phys: &Physics) {
        let mesh = crate::geometry::shapes::create_cube(&self.memory_allocator);
        let inst = Instance {
            model_matrix: Transform {
                position: pos,
                ..Default::default()
            }
            .to_matrix(),
            color: mat.color,
            velocity: phys.velocity,
            mass: phys.mass,
            collision: phys.collision,
            gravity: phys.gravity,
            ..Default::default()
        };
        self.scene.lock().unwrap().add_instance(mesh, inst);
    }

    pub fn add_sphere(&mut self, pos: [f32; 3], scale: f32, mat: &Material, phys: &Physics) {
        let mesh = crate::geometry::shapes::create_sphere_subdivided(&self.memory_allocator, 3);
        let inst = Instance {
            model_matrix: Transform {
                position: pos,
                scale: [scale, scale, scale],
                ..Default::default()
            }
            .to_matrix(),
            color: mat.color,
            velocity: phys.velocity,
            mass: phys.mass,
            collision: phys.collision,
            gravity: phys.gravity,
            ..Default::default()
        };
        self.scene.lock().unwrap().add_instance(mesh, inst);
    }

    pub fn run(mut self) {
        let mut previous_frame_end: Option<Box<dyn GpuFuture>> =
            Some(sync::now(self.base.device.clone()).boxed());

        let (physics_read, physics_write, mut solid_obj_count) = {
            let mut s = self.scene.lock().unwrap();
            s.upload_to_gpu(&self.memory_allocator, &self.base.queue);
            let tex_count = s.texture_views.len();
            s.ensure_descriptor_cache(&self.pipeline, tex_count);
            (
                s.physics_read.clone(),
                s.physics_write.clone(),
                s.total_instances,
            )
        };

        let compute_layout = self.compute_pipeline.layout().set_layouts()[0].clone();
        let set_0 = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            compute_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_read.clone()),
                WriteDescriptorSet::buffer(1, physics_write.clone()),
            ],
        )
        .unwrap();
        let set_1 = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            compute_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, physics_write.clone()),
                WriteDescriptorSet::buffer(1, physics_read.clone()),
            ],
        )
        .unwrap();

        let mut framebuffers =
            create_framebuffers(&self.images, &self.render_pass, &self.memory_allocator);
        let mut inputs = InputState::default();
        let mut last_frame_instant = Instant::now();
        let mut accumulator = 0.0;
        let fixed_dt = 1.0 / 60.0;
        let mut compute_ping_pong = false;
        let mut recreate_swapchain = false;
        let mut frame_index = 0;

        let start_time = Instant::now();
        let mut fps_timer = Instant::now();
        let mut frame_count = 0u32;

        let event_loop = self.event_loop.take().unwrap();
        event_loop.run(move |event, _, control_flow| {
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

                    frame_count += 1;
                    if fps_timer.elapsed().as_secs_f32() >= 2.0 {
                        println!(
                            "FPS: {:.0}",
                            frame_count as f32 / fps_timer.elapsed().as_secs_f32()
                        );
                        frame_count = 0;
                        fps_timer = Instant::now();
                    }

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

                    let physics_idx = {
                        let mut s = self.scene.lock().unwrap();
                        s.prepare_frame_ubo(frame_index, view, proj, cam_pos);
                        solid_obj_count = s.total_instances;
                        let tex_count = s.texture_views.len();
                        s.ensure_descriptor_cache(&self.pipeline, tex_count);
                        if compute_ping_pong {
                            0
                        } else {
                            1
                        }
                    };

                    let mut comp_builder =
                        create_builder(&self.command_buffer_allocator, &self.base.queue);
                    let mut physics_ran = false;
                    while accumulator >= fixed_dt {
                        let active_set = if compute_ping_pong { &set_1 } else { &set_0 };
                        record_compute_physics(
                            &mut comp_builder,
                            &self.compute_pipeline,
                            active_set,
                            solid_obj_count,
                            fixed_dt,
                            solid_obj_count,
                        );
                        compute_ping_pong = !compute_ping_pong;
                        accumulator -= fixed_dt;
                        physics_ran = true;
                    }

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
                            &self.pipeline,
                        );
                        s.record_draws(
                            &mut render_builder,
                            &self.pipeline,
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
