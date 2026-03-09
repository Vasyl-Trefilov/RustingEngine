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

mod shapes;
use shapes::{VertexPosColor, Mesh, Scene, SceneObject, Transform};
use shapes::shapes::{create_cube, create_sphere, create_plane};

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

struct RenderObject {
    mesh: Mesh,
    transform: Transform,
    uniform_buffer: Subbuffer<UniformBufferObject>,
    descriptor_set: Arc<PersistentDescriptorSet>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBufferObject {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}


fn create_render_object(
    _device: &Arc<Device>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    pipeline: &Arc<GraphicsPipeline>,
    mesh: Mesh,
    transform: Transform,
    dims: [u32; 2],
) -> RenderObject {
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians();
    let z_near = 0.1;
    let z_far = 100.0;
    let f = 1.0 / (fov / 2.0).tan();

    let proj = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (z_far + z_near) / (z_far - z_near), 1.0],
        [0.0, 0.0, -(2.0 * z_far * z_near) / (z_far - z_near), 0.0],
    ];

    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 5.0, 1.0],
    ];

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

    let descriptor_set = PersistentDescriptorSet::new(
        descriptor_set_allocator,
        pipeline.layout().set_layouts().get(0).unwrap().clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
    ).unwrap();

    RenderObject { mesh, transform, uniform_buffer, descriptor_set }
}

fn create_framebuffers(
    images: &[Arc<dyn ImageViewAbstract>],
    render_pass: &Arc<vulkano::render_pass::RenderPass>,
    memory_allocator: &StandardMemoryAllocator, 
    dims: [u32; 2]
) -> Vec<Arc<Framebuffer>> {
    
    let depth_image = AttachmentImage::transient(
        memory_allocator, 
        dims, 
        Format::D16_UNORM
    ).unwrap();

    let depth_image_view = ImageView::new_default(depth_image).unwrap();

    images.iter().map(|img: &Arc<dyn ImageViewAbstract>| {
        Framebuffer::new(render_pass.clone(), FramebufferCreateInfo {
            attachments: vec![img.clone(), depth_image_view.clone()],
            ..Default::default()
        }).unwrap()
    }).collect()
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

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 v_color;

            layout(set = 0, binding = 0) uniform UniformBufferObject {
                mat4 model;
                mat4 view;
                mat4 proj;
            } ubo;

            void main() {
                gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
                v_color = color;
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

            layout(location = 0) in vec3 v_color;
            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(v_color, 1.0);
            }
            
        "
    }
}

fn main() {
    // ! VULKAN INITIALIZATION 
    let library = VulkanLibrary::new().expect("No Vulkan driver found.");
    let required_extensions = vulkano_win::required_extensions(&library);

    let instance = Instance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().with_title("Vulkan Engine").build_vk_surface(&event_loop, instance.clone()).unwrap();
    let window = surface.object().unwrap().clone().downcast::<Window>().unwrap();

    let device_extensions = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };

    let (physical_device, queue_family_index) = instance.enumerate_physical_devices().unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties().iter().enumerate().position(|(i, q)| {
                q.queue_flags.intersects(QueueFlags::GRAPHICS) && p.surface_support(i as u32, &surface).unwrap_or(false)
            }).map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0, PhysicalDeviceType::IntegratedGpu => 1, _ => 2,
        }).unwrap();

    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
        ..Default::default()
    }).unwrap();
    let queue = queues.next().unwrap(); 

    // ! SWAPCHAIN: THE BACK BUFFER
    let (mut swapchain, images) = {
        let caps = device.physical_device().surface_capabilities(&surface, Default::default()).unwrap();
        let format = device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0].0;
        
        let (sw, img) = Swapchain::new(device.clone(), surface.clone(), SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_format: Some(format),
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: caps.supported_composite_alpha.into_iter().next().unwrap(),
            present_mode: PresentMode::Immediate,
            ..Default::default()
        }).unwrap();
        (sw, img)
    };

    let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = std::sync::Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()));
    // ! PIPELINE: THE DRAWING RECIPE 
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare, 
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [ {color: [color], depth_stencil: {depth}, input: []} ],
    ).unwrap();

    let image_views: Vec<Arc<dyn ImageViewAbstract>> = images.iter()
        .map(|img| {
            let view = ImageView::new_default(img.clone()).unwrap();
            view as Arc<dyn ImageViewAbstract>
        })
        .collect();
    let mut framebuffers = create_framebuffers(&image_views, &render_pass, &memory_allocator, window.inner_size().into());

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let vertex_input_state = VertexPosColor::per_vertex();
    let vertex_shader_entrypoint = vs.entry_point("main").unwrap();
    let fragment_shader_entrypoint = fs.entry_point("main").unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(VertexPosColor::per_vertex())
        .vertex_shader(vertex_shader_entrypoint, ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList)) // Changed to TriangleList for cubes
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fragment_shader_entrypoint, ())
        .depth_stencil_state(DepthStencilState::simple_depth_test()) 
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();


    // * Allocator for Command Buffers (the lists of tasks we send to the GPU).
    let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default());
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    
    let layout = pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let mut render_objects = Vec::new();

    // let dims = window.inner_size().into();
    let dims: [u32; 2] = [1920, 1080]; // Placeholder dimensions for object creation
    let cube_mesh = create_cube(&memory_allocator, [1.0, 0.0, 0.0]);
    render_objects.push(create_render_object(
        &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
        cube_mesh.clone(),
        Transform { position: [-1.0, 1.0, 0.0], ..Default::default() }, 
        dims
    ));
    render_objects.push(create_render_object(
        &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
        cube_mesh.clone(),
        Transform { position: [1.0, -1.0, 0.0], ..Default::default() }, 
        dims
    ));
    render_objects.push(create_render_object(
        &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
        cube_mesh.clone(),
        Transform { position: [-1.0, -1.0, 0.0], ..Default::default() }, 
        dims
    ));
    render_objects.push(create_render_object(
        &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
        cube_mesh,
        Transform { position: [1.0, 1.0, 0.0], ..Default::default() }, 
        dims
    ));

    // let sphere_mesh = create_sphere(&memory_allocator, [0.0, 0.0, 1.0], 32, 16);
    // render_objects.push(create_render_object(
    //     &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
    //     sphere_mesh,
    //     Transform { position: [-2.0, 0.0, -5.0], ..Default::default() }, 
    //     dims
    // ));

    // let plane_mesh = create_plane(&memory_allocator, [0.0, 1.0, 0.0], 5.0, 5.0);
    // render_objects.push(create_render_object(
    //     &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
    //     plane_mesh,
    //     Transform { position: [0.0, -2.0, -5.0], ..Default::default() }, 
    //     dims
    // ));

    let start_time = std::time::Instant::now(); 
    let mut frame_count: u32 = 0;
    let mut fps_timer = std::time::Instant::now();
    // ! MAIN RENDER LOOP - runs until the window is closed.
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll; 

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { *control_flow = ControlFlow::Exit; },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => { recreate_swapchain = true; },
            Event::MainEventsCleared => {
                let elapsed = start_time.elapsed().as_secs_f32();
                let dims: [u32; 2] = window.inner_size().into();
                if dims[0] == 0 || dims[1] == 0 { return; } 
                let aspect = dims[0] as f32 / dims[1] as f32;
                let fov = 45.0f32.to_radians();
                let z_near = 0.1;
                let z_far = 100.0;
                let f = 1.0 / (fov / 2.0).tan();

                let proj = [
                    [f / aspect, 0.0, 0.0, 0.0],
                    [0.0, f, 0.0, 0.0],
                    [0.0, 0.0, (z_far + z_near) / (z_far - z_near), 1.0],
                    [0.0, 0.0, -(2.0 * z_far * z_near) / (z_far - z_near), 0.0],
                ];
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
                if recreate_swapchain {
                    unsafe {
                        device.wait_idle().unwrap();
                    }

                    let (new_sw, new_img) = swapchain.recreate(SwapchainCreateInfo { image_extent: dims, ..swapchain.create_info() }).unwrap();
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
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let (img_index, suboptimal, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r, Err(AcquireError::OutOfDate) => { recreate_swapchain = true; return; }, Err(e) => panic!("{:?}", e),
                };
                if suboptimal { recreate_swapchain = true; }

                let mut builder = AutoCommandBufferBuilder::primary(&cb_allocator, queue.queue_family_index(), CommandBufferUsage::OneTimeSubmit).unwrap();
                
                
                builder.begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.1, 0.1, 0.2, 1.0].into()),
                            Some(1.0.into()),   
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