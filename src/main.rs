mod renderer;
mod scene;
mod input;
mod shaders;
mod shapes;
mod effects;

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
use crate::renderer::swapchain::{create_framebuffers, create_render_pass, create_swapchain_and_images};
use crate::shapes::VertexPosColorNormal;
 
use std::{panic};
use rand::*;
use crate::renderer::camera::{Camera, camera_rotate, create_look_at, create_projection_matrix};
use crate::scene::RenderScene;
use crate::input::{InputState, MouseState, set_mouse_capture};
use crate::scene::object::InstanceData;
use crate::shaders::vs;
use crate::shaders::fs;
use crate::scene::object::Instance;
use crate::scene::object::Transform;
use crate::renderer::pipeline::create_pipeline;
use crate::renderer::render::process_render;
use crate::renderer::render::create_builder;
use crate::shapes::gltfLoader::{load_gltf_scene};
use crate::shapes::shapes::{create_sphere_subdivided, create_triangle};
use crate::effects::{RainSettings, SphereSettings, create_event_horizon, create_fire, create_fountain, create_monochrome_rain, create_nebula_sphere, create_void_fire};
use crate::scene::InstanceHandle;

fn main() {
    let event_loop = EventLoop::new();
    let dims = [1920, 1080]; // Placeholder dimensions for projection matrix
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians();  // Field of view in radians, its like a minecraft fov, if you know
    let z_near = 0.1;                 // Near clipping plane, it means, if some object is 0.1 from camera, it will not be shown
    let z_far = 500.0;                 // Far clipping plane, how far can 'camera' see, you can set like 1000 if you are not developing some AAA game, but if you do, I guess you know better then me what to do

    // ! PROJECTION MATRIX - Converts 3D to 2D screen coordinates
    let mut proj: [[f32; 4]; 4] = create_projection_matrix(aspect, fov, z_near, z_far);
    // ! VIEW MATRIX - Camera position (currently looking from [0,0,5])
    let mut view: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 350.0, 1.0],
    ];
    let mut eye_pos: [f32; 3] = [view[3][0],view[3][1],view[3][2]];

    // Initialize Vulkan Base
    let base = renderer::init_vulkan(&event_loop);

    let cb_allocator = StandardCommandBufferAllocator::new(base.device.clone(), StandardCommandBufferAllocatorCreateInfo::default());
    
    let vs = vs::load(base.device.clone()).unwrap();
    let fs = fs::load(base.device.clone()).unwrap();

    let (mut swapchain, images) = create_swapchain_and_images(&base.device, &base.surface, &base.window);

    let render_pass = create_render_pass(base.device.clone(), &swapchain);
    // ! GRAPHICS PIPELINE - The complete configuration for drawing
    let pipeline = create_pipeline(vs, fs, &render_pass, &base.device);

    let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = 
        std::sync::Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(base.device.clone()));
    let descriptor_set_allocator: Arc<StandardDescriptorSetAllocator> = Arc::new(StandardDescriptorSetAllocator::new(base.device.clone()));

    // * Inputs
    let mut mouse_state = MouseState::default();
    // let mut prev_mouse_state = MouseState::default();
    let mut inputs = InputState{speed: 0.01, ..Default::default()};

    // * Scene
    let mut scene = RenderScene::new(&memory_allocator, &descriptor_set_allocator, &pipeline, &base.queue, 3, 1_000_000);
    let mut camera = Camera {
        position: [0.0, 0.0, 10.0],
        yaw: -90.0f32.to_radians(),
        pitch: 0.0,
    };
    scene.set_light([20.0, 20.0, 20.0], [1.0, 1.0, 1.0], 10000.0);
    let mut rng = rand::rng();
    let triangle = create_triangle(&memory_allocator, [1.0,1.0,1.0]);

    let (objects, textures) = load_gltf_scene(&memory_allocator, "./testModels/Rustyball.gltf"); // ! 3D MODELS IMPORT, LETS GO
    scene.set_textures(&textures, &base.queue, &memory_allocator);
    for (mesh, instance) in objects {
        scene.add_instance(mesh, instance);
    }

    // create_star_sphere(&mut scene, triangle.clone(), 10000); // * this is the main performance check, just bc why not
    // create_fountain(&mut scene, triangle.clone(), 500);
    // create_fire(&mut scene, triangle.clone(), 4000, None);
    // create_void_fire(&mut scene, triangle.clone(), 3000, None);
    // create_nebula_sphere(&mut scene, triangle.clone(), 3000, None);
    // create_event_horizon(&mut scene, triangle.clone(), 3000, Some(SphereSettings{center: [0.0,0.0,0.0], radius: 20.0, random_color: true, ..Default::default()}));
    // create_monochrome_rain(&mut scene, triangle.clone(), 3000, Some(RainSettings{speed: 0.01, ..Default::default()}));
    
    let sphere_sub_mesh = create_sphere_subdivided(&memory_allocator, [1.0,1.0,1.0], 3);
    
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
    let mut effect = 0;
    let mut effect_handlers: Vec<InstanceHandle> = Vec::new();
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
                        inputs.is_mouse_dragging = state == winit::event::ElementState::Pressed;
                    }
                    if button == winit::event::MouseButton::Right {
                    //     effect_handlers.sort_by(|a, b| {
                    //         b.batch_index.cmp(&a.batch_index)
                    //             .then(b.instance_index.cmp(&a.instance_index))
                    //     });
                    //     for handle in &effect_handlers {
                    //         scene.remove_instance(*handle);
                    //     }
                    //     effect_handlers.clear();
                    //     match effect{
                    //         0=>effect_handlers = create_monochrome_rain(&mut scene, triangle.clone(), 3000, Some(RainSettings{speed: 0.01, ..Default::default()})),
                    //         1=>effect_handlers = create_fountain(&mut scene, triangle.clone(), 500, None),
                    //         2=>effect_handlers = create_fire(&mut scene, triangle.clone(), 4000, None),
                    //         3=>effect_handlers = create_void_fire(&mut scene, triangle.clone(), 3000, None),
                    //         4=>effect_handlers = create_nebula_sphere(&mut scene, triangle.clone(), 3000, None),
                    //         5=>effect_handlers = create_event_horizon(&mut scene, triangle.clone(), 3000, Some(SphereSettings{center: [0.0,0.0,0.0], radius: 20.0, random_color: true, ..Default::default()})),
                    //         _=>effect=-1
                    // } 
                    // effect+=1;
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

