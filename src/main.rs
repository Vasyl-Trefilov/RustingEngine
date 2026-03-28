mod effects;
mod geometry;
mod input;
mod rendering;
mod scene;
mod shaders;

use std::f32::consts::PI;
use std::fmt::format;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::image::ImageUsage;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::Pipeline;
use vulkano::swapchain::CompositeAlpha;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::{AcquireError, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::AccessFlags;
use vulkano::sync::PipelineStages;
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::NonExhaustive;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::geometry::VertexPosColorNormal;
use crate::rendering::swapchain::{
    create_framebuffers, create_render_pass, create_swapchain_and_images,
};
use crate::scene::animation::AnimationType;
use std::sync::Arc;

use crate::effects::{
    create_event_horizon, create_fire, create_fountain, create_monochrome_rain,
    create_nebula_sphere, create_void_fire, RainSettings, SphereSettings,
};
use crate::geometry::gltfLoader::load_gltf_scene;
use crate::geometry::shapes::{
    create_cube, create_plane, create_sphere_subdivided, create_triangle,
};
use crate::input::{set_mouse_capture, InputState, MouseState};
use crate::rendering::camera::{camera_rotate, create_look_at, create_projection_matrix, Camera};
use crate::rendering::pipeline::create_pipeline;
use crate::rendering::render::create_builder;
use crate::rendering::render::process_render;
use crate::scene::object::Instance;
use crate::scene::object::InstanceData;
use crate::scene::object::Transform;
use crate::scene::RenderScene;
use crate::scene::{begin_render_pass_only, record_compute_physics, InstanceHandle};
use crate::shaders::fs;
use crate::shaders::vs;
use rand::*;
use smallvec::SmallVec;
use std::{default, panic};
use vulkano::command_buffer::synced::SyncCommandBufferBuilder;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::sync::{BufferMemoryBarrier, DependencyInfo};

fn main() {
    let event_loop = EventLoop::new();
    let dims = [1920, 1080]; // Placeholder dimensions for projection matrix
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians(); // Field of view in radians, its like a minecraft fov, if you know
    let z_near = 0.1; // Near clipping plane, it means, if some object is 0.1 from camera, it will not be shown
    let z_far = 1000.0; // Far clipping plane, how far can 'camera' see, you can set like 1000 if you are not developing some AAA game, but if you do, I guess you know better then me what to do

    // ! PROJECTION MATRIX - Converts 3D to 2D screen coordinates
    let mut proj: [[f32; 4]; 4] = create_projection_matrix(aspect, fov, z_near, z_far);
    // ! VIEW MATRIX - Camera position (currently looking from [0,0,5])
    let mut view: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 350.0, 1.0],
    ];
    let mut eye_pos: [f32; 3] = [view[3][0], view[3][1], view[3][2]];

    // Initialize Vulkan Base
    let base = rendering::init_vulkan(&event_loop, "RustingEngine");

    let cb_allocator = StandardCommandBufferAllocator::new(
        base.device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let vs = vs::load(base.device.clone()).unwrap();
    let fs = fs::load(base.device.clone()).unwrap();

    let (mut swapchain, images) = Swapchain::new(
        base.device.clone(),
        base.surface.clone(),
        SwapchainCreateInfo {
            min_image_count: 3, // Triple buffering
            image_format: None,
            image_extent: dims,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
    )
    .unwrap();
    let render_pass = create_render_pass(base.device.clone(), &swapchain);
    // ! GRAPHICS PIPELINE - The complete configuration for drawing
    let pipeline = create_pipeline(vs, fs, &render_pass, &base.device);

    let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> =
        std::sync::Arc::new(
            vulkano::memory::allocator::StandardMemoryAllocator::new_default(base.device.clone()),
        );
    let descriptor_set_allocator: Arc<StandardDescriptorSetAllocator> =
        Arc::new(StandardDescriptorSetAllocator::new(base.device.clone()));

    // * Inputs
    let mut mouse_state = MouseState::default();
    // let mut prev_mouse_state = MouseState::default();
    let mut inputs = InputState {
        speed: 0.1,
        ..Default::default()
    };
    let mut rng = rand::rng();
    let triangle = create_triangle(&memory_allocator);
    // let plane = create_plane(&memory_allocator, [0.0,0.0,0.0], 10.0, 10.0);
    let cube = create_cube(&memory_allocator);
    // * Scene
    let mut scene = RenderScene::new(
        &memory_allocator,
        &descriptor_set_allocator,
        &pipeline,
        &base.queue,
        3,
        1_000_000,
    ); // 1_000_000
    let mut camera = Camera {
        position: [0.0, 5.0, 20.0],
        yaw: 90.0f32.to_radians(),
        pitch: 0.0,
    };
    scene.set_light([30.0, 30.0, 30.0], [1.0, 0.95, 0.9], 450.0);

    let (objects, textures) =
        load_gltf_scene(&memory_allocator, "./testModels/1kRustingSphere.gltf");
    scene.set_textures(&pipeline, &textures, &base.queue, &memory_allocator);

    // for (mesh, mut instance) in objects {
    //     instance.emissive = 0.0;
    //     instance.roughness = 0.85;
    //     instance.metalness = 0.15;

    //     instance.velocity = [0.0, 0.0, 0.0, 1.0];
    //     let mut transform;
    //     for i in 0..10 {
    //         for j in 0..10 {
    //             for k in 0..10 {
    //                 transform = Transform { position: [i as f32 * 2.0, j as f32 * 2.0, k as f32 * 2.0], ..Default::default() };
    //                 instance.model_matrix = transform.to_matrix();
    //                 scene.add_instance(mesh.clone(), instance.clone());
    //             }

    //         }
    //     }

    // }
    let mut position = [0.0, 0.0, 0.0];

    // scene.add_instance(cube.clone(), Instance {
    //     model_matrix: Transform {
    //         position: [2.2, 20.0, 0.0],
    //         scale: [5.0, 5.0, 5.0],
    //         ..Default::default()
    //     }.to_matrix(),
    //     velocity: [0.0, 0.0, 0.0, 2.5],
    //     mass: 100.0,
    //     collision: 0.0,
    //     gravity: 0.2,
    //     color: [1.0, 0.0, 0.0],
    //     emissive: 1.0,
    //     ..Default::default()
    // });

    // let sphere = create_sphere_subdivided(&memory_allocator, 4);
    // scene.add_instance(sphere.clone(), Instance {
    //     model_matrix: Transform{ position: [0.0, 2.5, 0.0], scale: [5.0, 5.0, 5.0], ..Default::default() }.to_matrix(),
    //     velocity: [0.0, 0.0, 0.0, 5.0],
    //     mass: 100.0,
    //     collision: 1.0,
    //     gravity: 0.0, /
    //     ..Default::default()
    // });

    // scene.add_instance(sphere.clone(), Instance {
    //     model_matrix: Transform{ position: [0.0, 15.0, 0.0], ..Default::default() }.to_matrix(),
    //     velocity: [0.0, 0.0, 0.0, 2.5], // Radius 2.5
    //     mass: 100.0,
    //     collision: 1.0,
    //     gravity: 0.2,
    //     ..Default::default()
    // });

    // scene.add_instance(triangle.clone(), Instance {
    //     model_matrix: Transform {
    //         position: [0.0, 0.0, 0.0],
    //         scale: [500.0, 500.0, 500.0],
    //         rotation: [PI/2.0, 0.0, 0.0],
    //         ..Default::default()
    //     }.to_matrix(),
    //     collision: 5.0,
    //     ..Default::default()
    // });

    for i in 0..10 {
        for j in 0..3 {
            for k in 0..10 {
                let pos = [i as f32 * 1.5, j as f32 * 2.0 + 7.0, k as f32 * 1.5];
                scene.add_instance(
                    cube.clone(),
                    Instance {
                        velocity: [0.0, 0.0, 0.0, 0.5],
                        model_matrix: Transform {
                            position: pos,
                            ..Default::default()
                        }
                        .to_matrix(),
                        mass: 1.0,
                        collision: 1.0,
                        gravity: 0.0,
                        color: [1.0, 0.0, 0.0],
                        ..Default::default()
                    },
                );
            }
        }
    }

    let sphere_sub_mesh = create_sphere_subdivided(&memory_allocator, 3);
    scene.add_instance(
        sphere_sub_mesh.clone(),
        Instance {
            velocity: [0.0, 0.0, 0.0, 7.5],
            model_matrix: Transform {
                position: [7.0, 100.0, 7.0],
                scale: [7.5, 7.5, 7.5],
                ..Default::default()
            }
            .to_matrix(),
            mass: 100000.0,
            collision: 1.0,
            gravity: 1.0,
            color: [0.0, 1.0, 0.0],
            ..Default::default()
        },
    );

    // let handle = scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [10.0, 20.0, 10.0], scale: [4.0, 4.0, 4.0], ..Default::default() },
    //         color: [1.0, 1.0, 0.0],
    //         emissive: 1.0,
    //         velocity: [0.0, 0.0, 0.0, 4.0],
    //         model_matrix: Transform { position: [30.0, 30.0, 30.0], scale: [4.0, 4.0, 4.0], ..Default::default() }.to_matrix(),
    //         ..Default::default()
    //     }
    // );
    // scene.add_instance(cube,
    //     Instance {
    //         transform: Transform { position: [0.0, 0.0, 0.0], scale: [4.0, 4.0, 4.0], ..Default::default() },
    //         color: [1.0, 1.0, 0.0],
    //         emissive: 1.0,
    //         velocity: [0.0, 0.0, 0.0, 4.0],
    //         model_matrix: Transform { position: [10.0, 10.0, 10.0], scale: [4.0, 4.0, 4.0], rotation: [PI/2.0, 0.0,0.0],  ..Default::default() }.to_matrix(),
    //         ..Default::default()
    //     }
    // );

    // let stars_logic = AnimationType::Custom(Arc::new(|transform, _velocity, original_pos, color, elapsed| {
    //     let speed = 0.1;
    //     let angle = elapsed * speed;

    //     let cos_a = angle.cos();
    //     let sin_a = angle.sin();

    //     transform.position[0] = original_pos[0] * cos_a - original_pos[2] * sin_a;
    //     transform.position[2] = original_pos[0] * sin_a + original_pos[2] * cos_a;

    // }));

    // for _ in 0..100000 {
    //     let radius = 100.0;

    //     let theta = rng.random_range(0.0..std::f32::consts::TAU);
    //     let phi = rng.random_range(0.0..std::f32::consts::PI);

    //     let x = radius * phi.sin() * theta.cos();
    //     let y = radius * phi.sin() * theta.sin();
    //     let z = radius * phi.cos();
    //     // let color = [rng.random_range(0.0..1.0),rng.random_range(0.0..1.0),rng.random_range(0.0..1.0)]; // This is easy and cool, but I am more dark/white guy(I mean, I like blackwhite style)
    //     let color = [1.0,1.0,1.0];
    //     scene.add_instance(
    //         triangle.clone(),
    //         Instance {
    //             transform: Transform {
    //                 position: [x, y, z],
    //                 scale: [0.2, 0.2, 0.2],
    //                 ..Default::default()
    //             },
    //             original_position: [x, y, z],
    //             animation: stars_logic.clone(),
    //             velocity: [0.0, 0.0, 0.0],
    //             color: color,
    //             emissive: 1.0,
    //             ..Default::default()
    //         }
    //     );
    // }
    // use crate::effects::FireSettings;
    // create_star_sphere(&mut scene, triangle.clone(), 10000); // * this is the main performance check, just bc why not
    // create_fountain(&mut scene, triangle.clone(), 500);
    // create_fire(&mut scene, triangle.clone(), 4000, Some(FireSettings{position: [20.0, 20.0, 20.0], max_height: 10.0, spread: 1.0}));
    // create_void_fire(&mut scene, triangle.clone(), 3000, None);
    // create_nebula_sphere(&mut scene, triangle.clone(), 3000, None);
    // create_event_horizon(&mut scene, triangle.clone(), 3000, Some(SphereSettings{center: [20.0,20.0,20.0], radius: 8.0, random_color: false, ..Default::default()}));

    // let rain_mesh = create_cube(&memory_allocator);
    // let rain_handles = create_monochrome_rain(
    //     &mut scene,
    //     triangle,
    //     2000,
    //     Some(RainSettings {
    //         area: [50.0, 100.0, 50.0],
    //         speed: 20.0,
    //         ..Default::default()
    //     })
    // );

    // * So I dont want to lie, this spheres was created by gemini, bc why not?
    // // 1. POLISHED COPPER (High Metalness + Low Roughness)
    // let handle = scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [-6.0, 0.0, 0.0], ..Default::default() },
    //         color: [0.89, 0.47, 0.33],
    //         shininess: 50.0,                 // Medium-sharp highlight
    //         specular_strength: 0.8,          // Strong reflection
    //         roughness: 0.05,                 // Very smooth surface, mirror-like
    //         metalness: 1.0,                  // 100% metal: light is tinted by the copper color
    //         ..Default::default()
    //     }
    // );

    // scene.remove_instance(handle);

    // // 2. CHROME / MIRROR (Pure White + Zero Roughness + Extreme Shininess)
    // scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [-3.6, 0.0, 0.0], ..Default::default() },
    //         color: [0.97, 0.97, 0.98],
    //         roughness: 0.0,                  // Perfectly smooth
    //         metalness: 1.0,                  // Reflects light source perfectly
    //         ..Default::default()
    //     }
    // );

    // // 3. 24K GOLD (Yellow Tint + High Shininess)
    // scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [-1.2, 0.0, 0.0], ..Default::default() },
    //         color: [1.0, 0.85, 0.4],
    //         shininess: 400.0,                // Very sharp highlight
    //         specular_strength: 0.9,
    //         roughness: 0.1,                  // Slight micro-scratches
    //         metalness: 1.0,                  // Metal tints specular highlights to gold
    //         ..Default::default()
    //     }
    // );

    // // 4. MATTE PLASTIC (Zero Metalness + High Roughness)
    // scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [1.2, 0.0, 0.0], ..Default::default() },
    //         color: [0.1, 0.4, 0.8],
    //         shininess: 5.0,                  // Very broad, dull highlight
    //         specular_strength: 0.1,          // Weak reflection
    //         roughness: 0.8,                  // Rough surface scatters light (no shine)
    //         metalness: 0.0,                  // Non-metal: uses standard diffuse lighting
    //         ..Default::default()
    //     }
    // );

    // // 5. GLOSSY CAR PAINT (Low Metalness + Very Low Roughness)
    // scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [3.6, 0.0, 0.0], ..Default::default() },
    //         color: [0.8, 0.05, 0.05],
    //         shininess: 600.0,                // Sharp reflection "clear coat" look
    //         specular_strength: 0.5,
    //         roughness: 0.02,                 // Very smooth finish
    //         metalness: 0.0,                  // Non-metal: white highlights on red base
    //         ..Default::default()
    //     }
    // );

    // // 6. BRUSHED ALUMINUM (High Metalness + High Roughness)
    // scene.add_instance(
    //     sphere_sub_mesh.clone(),
    //     Instance {
    //         transform: Transform { position: [6.0, 0.0, 0.0], ..Default::default() },
    //         color: [0.4, 0.42, 0.45],
    //         shininess: 20.0,                 // Wide, spread-out highlight
    //         specular_strength: 0.3,
    //         roughness: 0.7,                  // High roughness blurs the metallic reflection
    //         metalness: 1.0,                  // Still metal, but "satin" or "brushed" finish
    //         ..Default::default()
    //     }
    // );

    scene.upload_to_gpu(&memory_allocator, &base.queue);
    scene.ensure_descriptor_cache(&pipeline, textures.len());
    let solid_object_count = scene.total_instances;
    println!("{:?}", solid_object_count);
    let compute_shader = shaders::cs::load(base.device.clone()).unwrap(); // pls recompile shader
    let compute_pipeline = ComputePipeline::new(
        base.device.clone(),
        compute_shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap();

    let compute_layout = compute_pipeline.layout().set_layouts()[0].clone();
    println!("{:#?}", compute_layout.bindings());
    let mut compute_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        compute_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, scene.physics_read.clone()),
            WriteDescriptorSet::buffer(1, scene.physics_write.clone()),
        ],
    )
    .unwrap();

    let mut framebuffers = create_framebuffers(&images, &render_pass, &memory_allocator);

    let mut previous_frame_end: Option<Box<dyn GpuFuture>> =
        Some(vulkano::sync::now(base.device.clone()).boxed());
    let mut recreate_swapchain = false;
    let mut frame_index = 3;
    // let start_time = std::time::Instant::now();
    let mut dims: [u32; 2] = base.window.inner_size().into();
    let start_time = std::time::Instant::now();
    let mut frame_count: u32 = 0;
    let mut fps_timer = std::time::Instant::now();
    let mut total_fps = 0;
    let mut effect = 0;
    let mut effect_handlers: Vec<InstanceHandle> = Vec::new();

    // physic render
    let mut last_frame_instant = std::time::Instant::now();
    let mut accumulator = 0.0;
    let fixed_dt = 1.0 / 60.0;

    // For physic swap
    let set_0 = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        compute_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, scene.physics_read.clone()),
            WriteDescriptorSet::buffer(1, scene.physics_write.clone()),
        ],
    )
    .unwrap();

    let set_1 = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        compute_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, scene.physics_write.clone()),
            WriteDescriptorSet::buffer(1, scene.physics_read.clone()),
        ],
    )
    .unwrap();

    let mut compute_ping_pong = false;

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
            Event::WindowEvent {
                event: WindowEvent::CursorLeft { .. },
                ..
            } => {
                mouse_state.inside_window = false;
            }
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if inputs.mouse_captured {
                    let sensitivity = 0.001;
                    camera.yaw -= delta.0 as f32 * sensitivity;
                    camera.pitch += delta.1 as f32 * sensitivity;

                    camera.pitch = camera.pitch.clamp(-1.5, 1.5);
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorEntered { .. },
                ..
            } => {
                mouse_state.inside_window = true;
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(code) = input.virtual_keycode {
                        if input.state == winit::event::ElementState::Pressed {
                            if code == winit::event::VirtualKeyCode::Escape {
                                inputs.mouse_captured = !inputs.mouse_captured;
                                set_mouse_capture(&base.window, inputs.mouse_captured);
                            }
                            if code == winit::event::VirtualKeyCode::LShift {
                                inputs.sprint = 2.0;
                            }
                            inputs.keys_pressed.insert(code);
                        } else {
                            if code == winit::event::VirtualKeyCode::LShift {
                                inputs.sprint = 1.0;
                            }
                            inputs.keys_pressed.remove(&code);
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == winit::event::MouseButton::Left {
                        inputs.is_mouse_dragging = state == winit::event::ElementState::Pressed;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if !inputs.mouse_captured {
                        let dx = position.x as f32 - inputs.last_mouse_pos[0];
                        let dy = position.y as f32 - inputs.last_mouse_pos[1];
                        inputs.last_mouse_pos = [position.x as f32, position.y as f32];

                        if inputs.is_mouse_dragging {
                            let sensitivity = 0.001;
                            camera.yaw += dx * sensitivity;
                            camera.pitch += dy * sensitivity;
                            camera.pitch = camera.pitch.clamp(-1.5, 1.5);
                        }
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                frame_index = (frame_index + 1) % 3;

                let elapsed = start_time.elapsed().as_secs_f32();
                frame_count += 1;
                let elapsed_fps = fps_timer.elapsed().as_secs_f32();

                if elapsed_fps >= 2.0 {
                    let fps = frame_count as f32 / elapsed_fps;
                    println!("FPS: {:.0}", fps);
                    total_fps += frame_count;
                    frame_count = 0;
                    fps_timer = std::time::Instant::now();
                    if elapsed >= 10.0 {
                        println!("middle Fps: {:.0}", (total_fps as f32 / elapsed));
                    }
                }

                let now = std::time::Instant::now();
                let mut delta_time = now.duration_since(last_frame_instant).as_secs_f32();
                last_frame_instant = now;

                if delta_time > 0.05 {
                    delta_time = 0.05;
                }
                accumulator += delta_time;

                if recreate_swapchain {
                    let new_size = base.window.inner_size();
                    dims = [new_size.width, new_size.height];

                    let (new_sw, new_img) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: dims,
                            ..swapchain.create_info()
                        })
                        .unwrap();

                    swapchain = new_sw;
                    framebuffers = rendering::swapchain::create_framebuffers(
                        &new_img,
                        &render_pass,
                        &memory_allocator,
                    );

                    let aspect = dims[0] as f32 / dims[1] as f32;
                    proj = create_projection_matrix(aspect, fov, z_near, z_far);

                    recreate_swapchain = false;
                }

                let (img_index, suboptimal, acquire_future) =
                    match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("{e}"),
                    };

                view = camera_rotate(&mut camera, &inputs);
                eye_pos = camera.position;
                scene.prepare_frame_ubo(frame_index, view, proj, eye_pos);

                let mut builder = create_builder(&cb_allocator, &base.queue);

                while accumulator >= fixed_dt {
                    let active_compute_set = if compute_ping_pong { &set_1 } else { &set_0 };
                    record_compute_physics(
                        &mut builder,
                        &compute_pipeline,
                        active_compute_set,
                        scene.total_instances,
                        fixed_dt,
                        solid_object_count,
                    );
                    compute_ping_pong = !compute_ping_pong;
                    accumulator -= fixed_dt;
                }

                let compute_command_buffer = builder.build().unwrap();

                let compute_future = sync::now(base.device.clone())
                    .then_execute(base.queue.clone(), compute_command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();

                compute_future.wait(None).unwrap(); // Wait for compute to finish

                let mut render_builder = create_builder(&cb_allocator, &base.queue);

                // Ensure descriptor cache is ready
                let tex_count = scene.texture_views.len();
                scene.ensure_descriptor_cache(&pipeline, tex_count);

                begin_render_pass_only(
                    &mut render_builder,
                    &framebuffers,
                    img_index,
                    dims,
                    &pipeline,
                );
                let physics_idx = if compute_ping_pong { 0 } else { 1 };
                scene.record_draws(&mut render_builder, &pipeline, frame_index, physics_idx);
                render_builder.end_render_pass().unwrap();

                let render_command_buffer = render_builder.build().unwrap();

                let future = sync::now(base.device.clone())
                    .join(acquire_future)
                    .then_execute(base.queue.clone(), render_command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        base.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), img_index),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(f) => {
                        previous_frame_end = Some(sync::now(base.device.clone()).boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(base.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Flush error: {:?}", e);
                        previous_frame_end = Some(sync::now(base.device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}
