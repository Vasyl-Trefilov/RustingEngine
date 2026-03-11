//main.rs
// ! MAIN ENTRY POINT - This file contains the core Vulkan rendering engine
// * Vulkan is a low-level graphics API that gives us full control over the GPU

// Import all the Vulkan functionality we need
use vulkano::pipeline::Pipeline;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::VulkanLibrary;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use vulkano::swapchain::PresentMode;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, MemoryTypeFilter};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::buffer::Subbuffer;
use std::f32::consts::PI;
use std::sync::Arc;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::image::sys::Image;
use vulkano::image::sys::ImageCreateInfo;
use vulkano::image::ImageType;
use vulkano::format::Format;
use vulkano::image::ImageUsage;
use vulkano::image::{StorageImage, AttachmentImage};
use vulkano::device::DeviceOwned;
use vulkano::memory::allocator::MemoryAllocator; 
use vulkano::image::view::ImageViewAbstract;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::command_buffer::allocator::CommandBufferAllocator; 
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::rasterization::PolygonMode;
use vulkano::pipeline::graphics::color_blend::{ColorBlendState, ColorBlendAttachmentState};

// Import our custom shape and mesh system
mod shapes;
use shapes::{VertexPosColor, Mesh, Scene, SceneObject, Transform};
use shapes::shapes::{create_cube, create_sphere, create_plane};

// ! STAR STRUCTURE - For future particle system (currently unused)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Star {
    position: [f32; 3]  // 3D position of star
}

// ! PUSH CONSTANTS - Small data we can push directly to shaders (fast!)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    time: f32,      // Current time for animations
    aspect: f32,    // Screen aspect ratio (width/height)
}

// * RenderObject - Represents something we can draw on screen
struct RenderObject {
    mesh: Mesh,                                     // The actual geometry data
    transform: Transform,                           // Position/rotation/scale
    uniform_buffer: Subbuffer<UniformBufferObject>, // GPU buffer with transformation matrices
    descriptor_set: Arc<PersistentDescriptorSet>,   // Tells GPU where to find our uniform buffer
}

// ! UNIFORM BUFFER OBJECT - Matrix data sent to GPU each frame
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

// * Creates a render object with its own uniform buffer and descriptor set
fn create_render_object(
    _device: &Arc<Device>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    pipeline: &Arc<GraphicsPipeline>,
    mesh: Mesh,
    transform: Transform,
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4]
) -> RenderObject {

    // * Create GPU-side buffer for our uniform data
    let uniform_buffer = Buffer::from_data(
        &memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
        AllocationCreateInfo { usage: MemoryUsage::Upload, ..Default::default() },
        UniformBufferObject {
            model: transform.to_matrix(),
            view,
            proj,
        },
    ).unwrap();

    // * Descriptor set tells the shader "your uniform buffer is at this address"
    let descriptor_set = PersistentDescriptorSet::new(
        descriptor_set_allocator,
        pipeline.layout().set_layouts().get(0).unwrap().clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
    ).unwrap();

    RenderObject { mesh, transform, uniform_buffer, descriptor_set }
}

// * Creates framebuffers for each swapchain image (each needs its own color + depth buffer)
fn create_framebuffers(
    images: &[Arc<dyn ImageViewAbstract>],
    render_pass: &Arc<vulkano::render_pass::RenderPass>,
    memory_allocator: &StandardMemoryAllocator, 
    dims: [u32; 2]
) -> Vec<Arc<Framebuffer>> {
    
    // ! DEPTH BUFFER - Prevents far objects from drawing over near ones
    let depth_image = AttachmentImage::transient(
        memory_allocator, 
        dims, 
        Format::D16_UNORM  // 16-bit depth format
    ).unwrap();

    let depth_image_view = ImageView::new_default(depth_image).unwrap();

    // Create framebuffer for each swapchain image
    images.iter().map(|img: &Arc<dyn ImageViewAbstract>| {
        Framebuffer::new(render_pass.clone(), FramebufferCreateInfo {
            attachments: vec![img.clone(), depth_image_view.clone()], // Color + depth
            ..Default::default()
        }).unwrap()
    }).collect()
}

// ==========================================
// ! SHADERS: THE CODE THAT RUNS ON THE GPU
// ==========================================

// ! Vertex Shader runs once for every "point" (vertex) we draw.
// It transforms the vertex from 3D space to screen space
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
 #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;
            layout(location = 2) in vec3 barycentric; 

            layout(location = 0) out vec3 v_color;
            layout(location = 1) out vec3 v_barycentric; 

            layout(set = 0, binding = 0) uniform UniformBufferObject {
                mat4 model; mat4 view; mat4 proj;
            } ubo;

            void main() {
                gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
                v_color = color;
                v_barycentric = barycentric; 
            }
        "
    }
}

// ! Fragment Shader runs for every single pixel covered by our shape.
// It determines the final color of each pixel
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) in vec3 v_color;
            layout(location = 1) in vec3 v_barycentric; 

            layout(location = 0) out vec4 f_color;

            void main() {
                float min_dist = min(v_barycentric.x, min(v_barycentric.y, v_barycentric.z));
                if (min_dist < 0.01) {
                    f_color = vec4(1.0, 1.0, 1.0, 1.0);
                } else {
                    discard;
                    // f_color = vec4(v_color,0.0);
                }
            }
            
            
        "
    }
}

// ! MAIN FUNCTION - Program entry point
fn main() {
    let dims = [1920, 1080]; // Placeholder dimensions for projection matrix
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians();  // Field of view in radians
    let z_near = 0.1;                 // Near clipping plane
    let z_far = 100.0;                 // Far clipping plane
    let f = 1.0 / (fov / 2.0).tan();   // Focal length calculation

    // ! PROJECTION MATRIX - Converts 3D to 2D screen coordinates
    let proj: [[f32; 4]; 4] = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (z_far + z_near) / (z_far - z_near), 1.0],
        [0.0, 0.0, -(2.0 * z_far * z_near) / (z_far - z_near), 0.0],
    ];

    // ! VIEW MATRIX - Camera position (currently looking from [0,0,5])
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 5.0, 1.0],
    ];
    // ! VULKAN INITIALIZATION - Setting up the connection to the GPU
    let library = VulkanLibrary::new().expect("No Vulkan driver found.");
    let required_extensions = vulkano_win::required_extensions(&library);

    // * Create Vulkan instance (represents the Vulkan library state)
    let instance = Instance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).unwrap();

    // * Create window and surface (surface = window's drawing area)
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().with_title("Vulkan Engine").build_vk_surface(&event_loop, instance.clone()).unwrap();
    let window = surface.object().unwrap().clone().downcast::<Window>().unwrap();

    // * We need swapchain extension to show images on screen
    let device_extensions = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };

    // ! SELECT GPU - Find the best graphics card that can render to our window
    let (physical_device, queue_family_index) = instance.enumerate_physical_devices().unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties().iter().enumerate().position(|(i, q)| {
                q.queue_flags.intersects(QueueFlags::GRAPHICS) && p.surface_support(i as u32, &surface).unwrap_or(false)
            }).map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,  // Prefer discrete GPU
            PhysicalDeviceType::IntegratedGpu => 1, // Then integrated
            _ => 2,                                   // Anything else last
        }).unwrap();

    // ! CREATE LOGICAL DEVICE - Our handle to the GPU with enabled features
    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
        ..Default::default()
    }).unwrap();
    let queue = queues.next().unwrap();  // Get the command queue for submitting work

    // ! SWAPCHAIN: THE BACK BUFFER - Handles presenting images to screen
    let (mut swapchain, images) = {
        let caps = device.physical_device().surface_capabilities(&surface, Default::default()).unwrap();
        let format = device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0].0;
        
        let (sw, img) = Swapchain::new(device.clone(), surface.clone(), SwapchainCreateInfo {
            min_image_count: caps.min_image_count,  // Minimum buffers for smooth rendering
            image_format: Some(format),
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,  // We'll draw to these images
            composite_alpha: caps.supported_composite_alpha.into_iter().next().unwrap(),
            present_mode: PresentMode::Immediate,  // Show frames immediately (no vsync)
            ..Default::default()
        }).unwrap();
        (sw, img)
    };

    // * Memory allocator for creating GPU resources
    let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = 
        std::sync::Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()));
    
    // ! PIPELINE: THE DRAWING RECIPE - Defines how to process vertices and pixels
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {  // Color attachment (the final image)
                load: Clear,      // Clear to a color at start
                store: Store,     // Save the result
                format: swapchain.image_format(),
                samples: 1,       // No multisampling
            },
            depth: {  // Depth attachment (for 3D sorting)
                load: Clear,      // Clear to max depth
                store: DontCare,  // Don't need to save depth buffer
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [ {
            color: [color],        // Use color attachment
            depth_stencil: {depth}, // Use depth attachment
            input: []               // No input attachments
        } ],
    ).unwrap();

    // * Create image views (ways to interpret the image data) for each swapchain image
    let image_views: Vec<Arc<dyn ImageViewAbstract>> = images.iter()
        .map(|img| {
            let view = ImageView::new_default(img.clone()).unwrap();
            view as Arc<dyn ImageViewAbstract>
        })
        .collect();

    // * Create framebuffers (combine image views into render targets)
    let mut framebuffers = create_framebuffers(&image_views, &render_pass, &memory_allocator, window.inner_size().into());

    // * Load our shaders
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // ! GRAPHICS PIPELINE - The complete configuration for drawing
    let pipeline = GraphicsPipeline::start()
        // .color_blend_state(ColorBlendState::new(1).blend_alpha()) // Enable alpha blending
        .vertex_input_state(VertexPosColor::per_vertex())  // How vertices are laid out
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList)) // Draw triangles
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())  // We'll set viewport dynamically
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())  // Enable depth testing
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    // * Allocator for Command Buffers (the lists of tasks we send to the GPU)
    let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default());
    let mut recreate_swapchain = false;  // Flag for when window resizes
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());  // Track when GPU finishes
    
    // * Create descriptor set allocator for managing shader resource bindings
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    // ! CREATE SCENE OBJECTS 
    let mut render_objects = Vec::new();

    // let cube_mesh = create_cube(&memory_allocator, [0.0, 1.0, 0.0]);  // Green cube

    // render_objects.push(create_render_object(
    //     &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
    //     cube_mesh.clone(),
    //     Transform { position: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0], ..Default::default() }, 
    //     view, proj
    // ));

    let sphere_mesh = create_sphere(&memory_allocator, [0.0, 1.0, 0.0], 8, 16);  // Green sphere

    render_objects.push(create_render_object(
        &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
        sphere_mesh.clone(),
        Transform { position: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0], ..Default::default() }, 
        view, proj
    ));

    // render_objects.push(create_render_object(
    //     &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
    //     cube_mesh.clone(),
    //     Transform { position: [1.0, -1.0, 0.0], ..Default::default() }, 
    //     view, proj
    // ));
    // render_objects.push(create_render_object(
    //     &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
    //     cube_mesh.clone(),
    //     Transform { position: [-1.0, -1.0, 0.0], ..Default::default() }, 
    //     view, proj
    // ));
    // render_objects.push(create_render_object(
    //     &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
    //     cube_mesh,
    //     Transform { position: [1.0, 1.0, 0.0], ..Default::default() }, 
    //     view, proj
    // ));

    // * FPS counter setup
    let start_time = std::time::Instant::now(); 
    let mut frame_count: u32 = 0;
    let mut fps_timer = std::time::Instant::now();
    let mut time = 0.0f32;
    // ! MAIN RENDER LOOP - runs until the window is closed
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;  // Continuously check for events

        match event {
            // * Window events
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { 
                *control_flow = ControlFlow::Exit;  // Close window
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => { 
                recreate_swapchain = true;  // Window resized, need new swapchain
            },
            
            // ! RENDER FRAME - This is where the actual drawing happens
            Event::MainEventsCleared => {
                unsafe {
                    device.wait_idle().unwrap();
                }
                time += 0.002;
                let elapsed = start_time.elapsed().as_secs_f32();
                let dims: [u32; 2] = window.inner_size().into();
                if dims[0] == 0 || dims[1] == 0 { return; }  // Window minimized, skip rendering


                // * FPS calculation and display
                frame_count += 1;
                let elapsed_fps = fps_timer.elapsed().as_secs_f32();

                if elapsed_fps >= 1.0 {
                    let fps = frame_count as f32 / elapsed_fps;
                    println!("FPS: {:.0}", fps);
                    frame_count = 0;
                    fps_timer = std::time::Instant::now();
                }

                // ! RECREATE SWAPCHAIN if window was resized
                if recreate_swapchain {
                    unsafe {
                        device.wait_idle().unwrap();  // Wait for GPU to finish
                    }

                    let (new_sw, new_img) = swapchain.recreate(SwapchainCreateInfo { 
                        image_extent: dims, 
                        ..swapchain.create_info() 
                    }).unwrap();
                    swapchain = new_sw;
                    
                    let new_views: Vec<Arc<dyn ImageViewAbstract>> = new_img.iter()
                        .map(|img| {
                            let view = ImageView::new_default(img.clone()).unwrap();
                            view as Arc<dyn ImageViewAbstract> 
                        })
                        .collect();

                    framebuffers = create_framebuffers(new_views.as_slice(), &render_pass, &memory_allocator, dims);
                    recreate_swapchain = false;
                }

                // * Clean up finished GPU commands
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // ! ACQUIRE NEXT IMAGE from swapchain to draw on
                let (img_index, suboptimal, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => { 
                        recreate_swapchain = true; 
                        return; 
                    },
                    Err(e) => panic!("{:?}", e),
                };
                if suboptimal { recreate_swapchain = true; }

                // ! BUILD COMMAND BUFFER - Record all drawing commands
                let mut builder = AutoCommandBufferBuilder::primary(
                    &cb_allocator, 
                    queue.queue_family_index(), 
                    CommandBufferUsage::OneTimeSubmit
                ).unwrap();
                
                for obj in &mut render_objects {
                    obj.transform.rotation[1] = time;
                    obj.transform.rotation[2] = time;
                    obj.transform.rotation[0] = time;
                    let mut data = obj.uniform_buffer.write().unwrap();
                    data.model = obj.transform.to_matrix();
                }


                // * Start render pass (clearing color to dark blue and depth to 1.0)
                builder.begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()),  // Clear color
                            Some(1.0.into()),                    // Clear depth
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffers[img_index as usize].clone())
                    },
                    SubpassContents::Inline,
                ).unwrap()
                .set_viewport(0, vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dims[0] as f32, dims[1] as f32],
                    depth_range: 0.0..1.0,
                }])
                .bind_pipeline_graphics(pipeline.clone());

                // * Draw all objects
                for obj in &render_objects {
                    builder.bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        obj.descriptor_set.clone(),
                    )
                    .bind_vertex_buffers(0, (obj.mesh.vertices.clone(),)); 
                    
                    if let Some(indices) = &obj.mesh.indices {
                        builder.bind_index_buffer(indices.clone());
                        builder.draw_indexed(obj.mesh.index_count, 1, 0, 0, 0).unwrap();
                    } else {
                        builder.draw(obj.mesh.vertex_count, 1, 0, 0).unwrap();
                    }
                }

                builder.end_render_pass().unwrap();
                let command_buffer = builder.build().unwrap();

                // ! SUBMIT TO GPU - Send commands and present result
                let future = previous_frame_end.take().unwrap().join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), img_index))
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => previous_frame_end = Some(future.boxed()),
                    Err(FlushError::OutOfDate) => { 
                        recreate_swapchain = true; 
                        previous_frame_end = Some(sync::now(device.clone()).boxed()); 
                    },
                    Err(_) => previous_frame_end = Some(sync::now(device.clone()).boxed()),
                }
            },
            _ => ()
        }
    });
}