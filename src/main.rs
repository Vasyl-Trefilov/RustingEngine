use vulkano::pipeline::Pipeline;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::ImageUsage;
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
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::PipelineBindPoint;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Star {
    position: [f32; 3]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    time: f32,
    aspect: f32,
}

// ==========================================
// ! SHADERS: THE CODE THAT RUNS ON THE GPU
// ==========================================

// ! Vertex Shader runs once for every "point" (vertex) we draw.
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            // * define a 2D square made of 4 points.
            vec2 positions[4] = vec2[](
                vec2(-0.005, -0.005), vec2(0.005, -0.005),
                vec2(-0.005,  0.005), vec2(0.005,  0.005) // ! rescale to 0.5 if you want to test effect expect stars
            );
            // * add time to animate
            layout(push_constant) uniform PushConstants {
                float time;
                float aspect;
            } pc;
             
            layout(set = 0, binding = 0) buffer StarBuffer {
                vec3 starsPositions[];
            };

            // * this 'out' variable sends data to the Fragment Shader for every pixel.
            layout(location = 0) out vec2 v_uv;
            layout(location = 1) out float time;
            layout(location = 2) out float brightness;

            void main() {

            vec3 star_pos = starsPositions[gl_InstanceIndex];
            vec2 local_pos = positions[gl_VertexIndex];

            float depth = star_pos.z;

            float speed = 0.02 + depth * 0.08;

            float y = star_pos.y - pc.time * speed;

            float x = star_pos.x + sin(pc.time * 0.2 + star_pos.y * 10.0) * 0.02 * depth;

            y += sin(pc.time * 0.5 + star_pos.x * 8.0) * 0.02 * depth;

            y = fract(y * 0.5 + 0.5) * 2.0 - 1.0;

            vec2 pos = vec2(x, y);

            // * apply aspect correction to the entire coordinate system
            vec2 corrected = pos + local_pos;
            corrected.x *= pc.aspect;

            gl_Position = vec4(corrected, 0.0, 1.0);

            v_uv = local_pos;
            time = pc.time;
            brightness = star_pos.z;
        }
        "
    }
}

// ! Fragment Shader runs for every single pixel covered by our shape.
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec2 v_uv;
            layout(location = 1) in float time;
            layout(location = 2) in float brightness;

            layout(location = 0) out vec4 f_color;

            float random(vec2 st) {
                return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
            }

            void main() {

                // * creting a circle 
                // if (length(v_uv) > 0.5) { discard; }

                // * creating a ring by discarding the center
                // if (length(v_uv) < 0.4) { discard; }

                // * creating a ring with glow effect
                // f_color = vec4(exp(-500.0 * pow((length(v_uv) - 0.45), 2)), exp(-500.0 * pow((length(v_uv) - 0.45), 2)), 0.0, 1.0);

                // * ring without glow
                // f_color = vec4(1-length(v_uv)*2, 1-length(v_uv)*2, 1-length(v_uv)*2, 1.0);

                // * circle with white center and black edge
                // f_color = vec4(mix(vec3(1.0), vec3(0.0), length(v_uv) * 2.0), 1.0);

                // * animated color change from while to brack
                // f_color = vec4(mix(vec3(1.0), vec3(0.0), sin(time)), 1.0);

                // ! radar effect
                // * atan gives us the angle of the pixel relative to the center, which we can compare to out scanning line angle
                // float pixelAngle = atan(v_uv.y, v_uv.x);
                // * scanAngle is the angle of our scanning line, which rotates over time
                // float scanAngle = time * 2.0; 
                // * angleDiff is how far the pixel's angle is from the scanning line's angle, we use this to determine if the pixel should be bright
                // float angleDiff = abs(pixelAngle - scanAngle);
                // * I use mod to create a loop of angle from 0 to 2PI
                // angleDiff = mod(angleDiff, 2.0 * 3.14159); 
                // float lineWidth = 0.1; 
                // * smoothstep creates a soft edge for the scanning line, it returns 1.0 when angleDiff is 0 (pixel on line) and 0.0 when angleDiff is greater than lineWidth (pixel outside line)
                // float scanningLine = 1.0 - smoothstep(0.0, lineWidth, angleDiff);
                // * wave effect under the line
                // float wave = sin(length(v_uv) * 2.0 - time * 4.0) * 0.5 + 0.5;
                // * final color combines the wave, and scanning line effect
                // f_color = vec4(0.0,  wave * 0.5 + 0.5, scanningLine, 1.0);

                // ! learning SDF (signed distance function) to create shapes
                // * first we define 2 centers for circles
                //vec2 center1 = vec2(0.2, 0.2);
                //vec2 center2 = vec2(-0.2, -0.2);

                // * get distance to both
                //float dist1 = length(v_uv - center1);
                //float dist2 = length(v_uv - center2);

                // * now decide what to draw
                //if (dist1 < 0.1) { f_color = vec4(1.0, 0.0, 0.0, 1.0); } // * Red circle
                //else if (dist2 < 0.1) { f_color = vec4(0.0, 0.0, 1.0, 1.0); } // * Blue circle
                //else { discard; }

                // ! Starfield together with Rust's random function
                if (length(v_uv) > 0.005) discard;
                f_color = vec4(1.0, 1.0, 1.0, 1.0);
            }
        "
    }
}

fn main() {
    // ! VULKAN INITIALIZATION 
    // Load the Vulkan library from your OS (drivers).
    let library = VulkanLibrary::new().expect("No Vulkan driver found.");
    // Ask the windowing system what extensions Vulkan needs to talk to the screen.
    let required_extensions = vulkano_win::required_extensions(&library);
    
    // Create the 'Instance': the highest-level object in Vulkan.
    let instance = Instance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).unwrap();

    // Create the Window using winit.
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().with_title("Vulkan Engine").build_vk_surface(&event_loop, instance.clone()).unwrap();
    let window = surface.object().unwrap().clone().downcast::<Window>().unwrap();

    // We need 'khr_swapchain' to allow us to draw frames to the window.
    let device_extensions = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };

    // Select the best GPU (Prefer a discrete card over an integrated one).
    let (physical_device, queue_family_index) = instance.enumerate_physical_devices().unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // Ensure this GPU supports drawing (Graphics) and presenting to our specific window (Surface).
            p.queue_family_properties().iter().enumerate().position(|(i, q)| {
                q.queue_flags.intersects(QueueFlags::GRAPHICS) && p.surface_support(i as u32, &surface).unwrap_or(false)
            }).map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0, PhysicalDeviceType::IntegratedGpu => 1, _ => 2,
        }).unwrap();

    // Create the 'Device': the interface we use to send commands to the GPU.
    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
        ..Default::default()
    }).unwrap();
    let queue = queues.next().unwrap(); // A 'queue' is where we submit commands.

    // ! SWAPCHAIN: THE BACK BUFFER
    // The Swapchain holds the images that are being displayed on the monitor.
    let (mut swapchain, images) = {
        let caps = device.physical_device().surface_capabilities(&surface, Default::default()).unwrap();
        let format = device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0].0;
        Swapchain::new(device.clone(), surface.clone(), SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_format: Some(format),
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT, // We want to draw colors on it.
            composite_alpha: caps.supported_composite_alpha.into_iter().next().unwrap(),
            // present_mode: PresentMode::Immediate, // ! comment this line to use vsync
            ..Default::default()
        }).unwrap()
    };

    // ! PIPELINE: THE DRAWING RECIPE 
    // A RenderPass defines the "stages" of rendering (like clearing the screen).
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: { color: { load: Clear, store: Store, format: swapchain.image_format(), samples: 1, } },
        pass: { color: [color], depth_stencil: {} }
    ).unwrap();

    // Load our shaders into the GPU's memory.
    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // The GraphicsPipeline holds all state: shaders, topology, viewport, etc.
    let pipeline = GraphicsPipeline::start()
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone()).unwrap();

    // Create a Framebuffer for every image in the swapchain.
    let mut framebuffers = images.iter().map(|img| {
        Framebuffer::new(render_pass.clone(), FramebufferCreateInfo {
            attachments: vec![ImageView::new_default(img.clone()).unwrap()], ..Default::default()
        }).unwrap()
    }).collect::<Vec<_>>();

    // Allocator for Command Buffers (the lists of tasks we send to the GPU).
    let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default());
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    // stars for the starfield
    let stars: Vec<Star> = (0..100).map(|_| Star {
        position: [rand::random::<f32>() * 4.0 - 2.0, rand::random::<f32>() * 4.0 - 2.0, rand::random::<f32>() * 0.5 + 0.5]
    }).collect();
    let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = std::sync::Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()));
    let star_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { usage: MemoryUsage::Upload, ..Default::default() },
        stars.iter().copied()
    ).unwrap();
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, star_buffer.clone())]
    ).unwrap();
    // for animation timing
    let start_time = std::time::Instant::now(); 
    let mut frame_count: u32 = 0;
    let mut fps_timer = std::time::Instant::now();
    // ! MAIN RENDER LOOP - runs until the window is closed.
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll; // Run as fast as possible.

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { *control_flow = ControlFlow::Exit; },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => { recreate_swapchain = true; },
            Event::MainEventsCleared => {
                let elapsed = start_time.elapsed().as_secs_f32();
                let dims: [u32; 2] = window.inner_size().into();
                if dims[0] == 0 || dims[1] == 0 { return; } 
                let aspect = dims[1] as f32 / dims[0] as f32;
                let push = PushConstants {
                    time: elapsed,
                    aspect,
                };
                frame_count += 1;
                let elapsed_fps = fps_timer.elapsed().as_secs_f32();

                if elapsed_fps >= 1.0 {
                    let fps = frame_count as f32 / elapsed_fps;
                    println!("FPS: {:.0}", fps);

                    frame_count = 0;
                    fps_timer = std::time::Instant::now();
                }
                // If window resized, rebuild the swapchain.
                if recreate_swapchain {
                    let (new_sw, new_img) = swapchain.recreate(SwapchainCreateInfo { image_extent: dims, ..swapchain.create_info() }).unwrap();
                    swapchain = new_sw;
                    framebuffers = new_img.iter().map(|img| {
                        Framebuffer::new(render_pass.clone(), FramebufferCreateInfo { attachments: vec![ImageView::new_default(img.clone()).unwrap()], ..Default::default() }).unwrap()
                    }).collect::<Vec<_>>();
                    recreate_swapchain = false;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Get an image from the swapchain to draw on.
                let (img_index, suboptimal, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r, Err(AcquireError::OutOfDate) => { recreate_swapchain = true; return; }, Err(e) => panic!("{:?}", e),
                };
                if suboptimal { recreate_swapchain = true; }

                // Build the command buffer: "Clear screen, then draw."
                let mut builder = AutoCommandBufferBuilder::primary(&cb_allocator, queue.queue_family_index(), CommandBufferUsage::OneTimeSubmit).unwrap();
                
                builder.begin_render_pass(RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())], // ! Background color
                        ..RenderPassBeginInfo::framebuffer(framebuffers[img_index as usize].clone())
                    }, SubpassContents::Inline).unwrap()
                    .set_viewport(0, vec![Viewport { origin: [0.0, 0.0], dimensions: [dims[0] as f32, dims[1] as f32], depth_range: 0.0..1.0 }])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, descriptor_set.clone()) // ! CONNECT THE BUFFER
                    .push_constants(pipeline.layout().clone(), 0, push) // * send time and star positions to the shader
                    .draw(4, 100, 0, 0).unwrap() // Tells GPU to run the vertex shader 4 times and 100 instances (stars), and then run the fragment shader for every pixel covered by those vertices.
                    .end_render_pass().unwrap();
                
                let command_buffer = builder.build().unwrap();

                // Submit to the GPU.
                let future = previous_frame_end.take().unwrap().join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), img_index))
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => previous_frame_end = Some(future.boxed()),
                    Err(FlushError::OutOfDate) => { recreate_swapchain = true; previous_frame_end = Some(sync::now(device.clone()).boxed()); },
                    Err(_) => previous_frame_end = Some(sync::now(device.clone()).boxed()),
                }
            },
            _ => ()
        }
    });
}