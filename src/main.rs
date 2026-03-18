mod renderer;
mod scene;
mod input;
mod shaders;
mod shapes;

use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::swapchain::{SwapchainCreateInfo, SwapchainPresentInfo, AcquireError};
use vulkano::sync::{self, GpuFuture, FlushError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use vulkano::swapchain::Swapchain;
use vulkano::image::ImageUsage;
use vulkano::swapchain::PresentMode;
use crate::scene::animation::AnimationType;
use std::sync::Arc;
use crate::renderer::swapchain::{create_framebuffers, create_render_pass};
use crate::shapes::VertexPosColorNormal;
 
use std::{panic};
use rand::*;
use crate::renderer::camera::{Camera, camera_rotate, create_look_at, create_projection_matrix};
use crate::scene::RenderScene;
use crate::input::{InputState, MouseState, set_mouse_capture};
use crate::shapes::shapes::create_sphere_subdivided;
use crate::scene::object::InstanceData;
use crate::shaders::vs;
use crate::shaders::fs;
use crate::scene::object::Instance;
use crate::scene::object::Transform;
use crate::renderer::pipeline::create_pipeline;
use crate::renderer::render::process_render;
use crate::renderer::render::create_builder;
use crate::shapes::shapes::create_triangle;

fn main() {
    let event_loop = EventLoop::new();
    let dims = [1920, 1080]; // Placeholder dimensions for projection matrix
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians();  // Field of view in radians, its like a minecraft fov, if you know
    let z_near = 0.1;                 // Near clipping plane, it means, if some object is 0.1 from camera, it will not be shown
    let z_far = 500.0;                 // Far clipping plane, how far can 'camera' see, you can set like 1000 if you are not developing some AAA game, but if you do, I guess you know better then me what to do
    let f = 1.0 / (fov / 2.0).tan();   // Focal length calculation, yes, just google it if you need

    // ! PROJECTION MATRIX - Converts 3D to 2D screen coordinates
    let mut proj: [[f32; 4]; 4] = create_projection_matrix(aspect, fov, z_near, z_far);
    // ! VIEW MATRIX - Camera position (currently looking from [0,0,5])
    let mut view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 350.0, 1.0], // * This third value is camera position, but remeber, in OpenGL and THREE.js we use -Z to go back, in Vulkan we use +Z, so Z=350 in vulkan is Z=-350 in THREE.js/OpenGL
    ];
    let mut eye_pos = [view[3][0],view[3][1],view[3][2]];

    // Initialize Vulkan Base
    let base = renderer::init_vulkan(&event_loop);

    let cb_allocator = StandardCommandBufferAllocator::new(base.device.clone(), StandardCommandBufferAllocatorCreateInfo::default());
    
    let vs = vs::load(base.device.clone()).unwrap();
    let fs = fs::load(base.device.clone()).unwrap();

    let (mut swapchain, images) = {
        let caps = base.device.physical_device().surface_capabilities(&base.surface, Default::default()).unwrap();
        let format = base.device.physical_device().surface_formats(&base.surface, Default::default()).unwrap()[0].0;
        
        let (sw, img) = Swapchain::new(base.device.clone(), base.surface.clone(), SwapchainCreateInfo {
            min_image_count: caps.min_image_count,  // Minimum buffers for smooth rendering
            image_format: Some(format),
            image_extent: base.window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,  // We'll draw to these images
            composite_alpha: caps.supported_composite_alpha.into_iter().next().unwrap(),
            present_mode: PresentMode::Immediate,  // Show frames immediately (no vsync), can be used for benchmarking or just for fun
            // present_mode: PresentMode::Fifo, // So, only Fifo is guaranteed to be supported on every device. And I think its better for some kind of prod, if I ever will get to this point, but for now, I want to see the maximum fps, so I will use Immediate, but if you want to use Fifo, its your choise
            ..Default::default()
        }).unwrap();
        (sw, img)
    };

    let render_pass = create_render_pass(base.device.clone(), &swapchain);
    // ! GRAPHICS PIPELINE - The complete configuration for drawing
    let pipeline = create_pipeline(vs, fs, &render_pass, &base.device);

    let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = 
        std::sync::Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(base.device.clone()));
    let descriptor_set_allocator: StandardDescriptorSetAllocator = StandardDescriptorSetAllocator::new(base.device.clone());

    // * Inputs
    let mut mouse_state = MouseState::default();
    let mut prev_mouse_state = MouseState::default();
    let mut inputs = InputState::default();

    // * Scene
    let mut scene = RenderScene::new(&memory_allocator, &descriptor_set_allocator, &pipeline, 3, 100000);

    let mut rng = rand::rng();

    let stars_logic = AnimationType::Custom(Arc::new(|transform, _velocity, original_pos, elapsed| {
        let speed = 0.1; 
        let angle = elapsed * speed;
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        transform.position[0] = original_pos[0] * cos_a - original_pos[2] * sin_a;
        transform.position[2] = original_pos[0] * sin_a + original_pos[2] * cos_a;
        
    }));
    let triangle = create_triangle(&memory_allocator, [1.0,1.0,1.0]);
    for _ in 0..100000 {
        let radius = 100.0; 
        
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let phi = rng.random_range(0.0..std::f32::consts::PI);
        
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();
        // let color = [rng.random_range(0.0..1.0),rng.random_range(0.0..1.0),rng.random_range(0.0..1.0)]; // This is easy and cool, but I am more dark/white guy(I mean, I like blackwhite style)
        let color = [1.0,1.0,1.0];
        scene.add_instance(
            triangle.clone(),
            Instance {
                transform: Transform {
                    position: [x, y, z],
                    scale: [0.2, 0.2, 0.2],
                    ..Default::default()
                },
                original_position: [x, y, z], 
                animation: stars_logic.clone(),
                velocity: [0.0, 0.0, 0.0],
                color: color, 
                ..Default::default()
            }
        );
    }
    // scene.add_instance(
    //     create_sphere_subdivided(&memory_allocator, [1.0,1.0,1.0], 2), 
    //     Instance {
    //         transform: Transform { 
    //             position: [-4.0, 0.0, 0.0], 
    //             ..Default::default() 
    //         },
    //         color: [0.95, 0.6, 0.35],        
    //         shininess: 400.0,                 
    //         specular_strength: 1.0,          
    //         roughness: 0.05,                  
    //         metalness: 1.0,                  
    //         ..Default::default()
    //     }
    // );

    let mut camera = Camera {
        position: [0.0, 0.0, 350.0],
        yaw: -90.0f32.to_radians(),
        pitch: 0.0,
    };

    let mut framebuffers = create_framebuffers(&images, &render_pass, &memory_allocator);

    let mut previous_frame_end = Some(sync::now(base.device.clone()).boxed());
    let mut recreate_swapchain = false;
    let mut frame_index = 3;
    // let start_time = std::time::Instant::now(); 
    let mut dims: [u32; 2] = base.window.inner_size().into();
    let start_time = std::time::Instant::now(); 
    let mut frame_count: u32 = 0;
    let mut fps_timer = std::time::Instant::now();
    let mut total_fps = 0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
            Event::WindowEvent { event: WindowEvent::CursorLeft { .. }, .. } => {
                mouse_state.inside_window = false;
            },
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
            },
            Event::WindowEvent { event: WindowEvent::CursorEntered { .. }, .. } => {
                mouse_state.inside_window = true;
            },
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
                },
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == winit::event::MouseButton::Left {
                        inputs.is_mouse_dragging = (state == winit::event::ElementState::Pressed);
                    }
                },
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
                },
                _ => ()
            },
            Event::MainEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();
                let elapsed = start_time.elapsed().as_secs_f32();
                frame_count += 1;
                let elapsed_fps = fps_timer.elapsed().as_secs_f32();

                if elapsed_fps >= 2.0 {
                    let fps = frame_count as f32 / elapsed_fps;
                    println!("FPS: {:.0}", fps);
                    // println!("FPS: {:.0}", elapsed);
                    total_fps += frame_count;
                    frame_count = 0;
                    fps_timer = std::time::Instant::now();
                    if elapsed >= 10.0 {
                        println!("middle Fps: {:.0}", (total_fps as f32 / elapsed));
                    }
                }
                frame_index = (frame_index + 1) % 3;
                // 1. Handle Swapchain recreation
                if recreate_swapchain {
                    let new_size = base.window.inner_size(); 
                    dims = [new_size.width, new_size.height]; 

                    let (new_sw, new_img) = swapchain.recreate(SwapchainCreateInfo {
                        image_extent: dims, 
                        ..swapchain.create_info()
                    }).unwrap();
                    
                    swapchain = new_sw;
                    framebuffers = renderer::swapchain::create_framebuffers(&new_img, &render_pass, &memory_allocator);
                    
                    let aspect = dims[0] as f32 / dims[1] as f32;
                    proj = create_projection_matrix(aspect, fov, z_near, z_far);
                    
                    recreate_swapchain = false;
                }

                let (img_index, suboptimal, acquire_future) = 
                    match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => { recreate_swapchain = true; return; }
                        Err(e) => panic!("{e}"),
                    };

                if suboptimal { recreate_swapchain = true; }

                let mut builder = create_builder(&cb_allocator, &base.queue);
                process_render(&mut builder, &framebuffers, img_index, dims, &pipeline);

                view = camera_rotate(&mut camera, &inputs);
                eye_pos = camera.position;

                // Scene render, entry point for my future library
                // When I will release 1.0 version of library, I will try to avoid API changes
                scene.update(elapsed, &mouse_state);
                scene.render(&mut builder, &pipeline, &memory_allocator, frame_index, view, proj, eye_pos);

                builder.end_render_pass().unwrap();
                let command_buffer = builder.build().unwrap();

                // ! SUBMIT TO GPU - Send commands and present result
                let future = previous_frame_end.take().unwrap().join(acquire_future)
                    .then_execute(base.queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(base.queue.clone(), SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), img_index))
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => previous_frame_end = Some(future.boxed()),
                    Err(FlushError::OutOfDate) => { 
                        recreate_swapchain = true; 
                        previous_frame_end = Some(sync::now(base.device.clone()).boxed()); 
                    },
                    Err(_) => previous_frame_end = Some(sync::now(base.device.clone()).boxed()),
                }
            },
            _ => ()
        }
    });
}