mod renderer;
mod scene;
mod input;
mod shaders;
mod shapes;

use std::sync::Arc;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::{SwapchainCreateInfo, SwapchainPresentInfo, AcquireError};
use vulkano::sync::{self, GpuFuture, FlushError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use vulkano::pipeline::Pipeline;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance as OtherInstance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline};
use vulkano::buffer::{ BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo};
use vulkano::format::Format;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::render_pass::Subpass;
use vulkano_win::VkSurfaceBuild;
use winit::window::Window;
use vulkano::VulkanLibrary;
use winit::window::WindowBuilder;

use crate::shapes::VertexPosColorNormal;

use std::{default, panic};
use rand::prelude::*;
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::pipeline::graphics::vertex_input::{
    VertexInputBindingDescription, 
    VertexInputRate
};

use crate::renderer::camera::{Camera, create_look_at};
use crate::scene::RenderScene;
use crate::input::{InputState, MouseState};
use crate::shapes::shapes::create_sphere_subdivided;
use crate::scene::object::InstanceData;
// use crate::shaders::vs;
// use crate::shaders::fs;
use crate::scene::object::Instance;
use crate::scene::object::Transform;


// ! Vertex Shader runs once for every "point" (vertex) we draw.
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec3 normal;      
        layout(location = 3) in vec3 barycentric; 
        layout(location = 4) in vec4 model_row0;  // Instance data starts at 4, you can find it in GPU settings
        layout(location = 5) in vec4 model_row1;
        layout(location = 6) in vec4 model_row2;
        layout(location = 7) in vec4 model_row3;
        layout(location = 8) in vec3 instance_color; 
        layout(location = 9) in vec4 instance_mat_props;

        layout(location = 0) out vec3 v_color;
        layout(location = 1) out vec3 v_normal;
        layout(location = 2) out vec3 v_pos;
        layout(location = 3) out vec4 v_mat_data; 

        layout(set = 0, binding = 0) uniform UniformBufferObject {
            mat4 view;
            mat4 proj;
            vec3 eye_pos;
        } ubo;

        void main() {
            mat4 instance_model = mat4(model_row0, model_row1, model_row2, model_row3);
            
            vec4 world_pos = instance_model * vec4(position, 1.0);
            
            gl_Position = ubo.proj * ubo.view * world_pos;
            
            v_pos = world_pos.xyz; 
            
            v_color = instance_color; 
            
            v_normal = mat3(instance_model) * normal; 
            v_mat_data = instance_mat_props;
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
            layout(location = 1) in vec3 v_normal;
            layout(location = 2) in vec3 v_pos;
            layout(location = 3) in vec4 v_mat_data;

            layout(set = 0, binding = 0) uniform UniformBufferObject {
                mat4 view;
                mat4 proj;
                vec3 eye_pos; 
            } ubo;
            layout(location = 0) out vec4 f_color;

            void main() {
                float shininess = v_mat_data.x;
                float spec_strength = v_mat_data.y;
                float roughness = v_mat_data.z;
                float metalness = v_mat_data.w;

                vec3 light_pos = vec3(20.0, 20.0, 20.0); // A light in the sky, like a sun, I will add custom light later(sun, pointerLight and areaLight)
                vec3 light_color = vec3(1.0, 1.0, 1.0);

                vec3 norm = normalize(v_normal);
                vec3 light_dir = normalize(light_pos - v_pos);
                vec3 view_dir = normalize(ubo.eye_pos - v_pos);
                vec3 halfway_dir = normalize(light_dir + view_dir);

                float diff = max(dot(norm, light_dir), 0.0);
                vec3 diffuse = diff * light_color * (1.0 - metalness); 

                float ambient_strength = 0.05;
                vec3 ambient = ambient_strength * light_color * (1.0 - metalness);

                vec3 spec_color = mix(vec3(1.0), v_color, metalness); 
                
                float spec_angle = max(dot(norm, halfway_dir), 0.0);
                float spec_factor = pow(spec_angle, shininess) * (1.0 - roughness);
                vec3 specular = spec_strength * spec_factor * light_color * spec_color;

                vec3 result = (ambient + diffuse) * v_color + specular;
                
                f_color = vec4(result, 1.0);
                // f_color = vec4(vec3(v_mat_data.y), 1.0); // spec_strength test
                // f_color = vec4(1.0,1.0,1.0,1.0); // Debug, hardcoded white color for every pixel
            }
        "
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let dims = [1920, 1080]; // Placeholder dimensions for projection matrix
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians();  // Field of view in radians, its like a minecraft fov, if you know
    let z_near = 0.1;                 // Near clipping plane, it means, if some object is 0.1 from camera, it will not be shown
    let z_far = 500.0;                 // Far clipping plane, how far can 'camera' see, you can set like 1000 if you are not developing some AAA game, but if you do, I guess you know better then me what to do
    let f = 1.0 / (fov / 2.0).tan();   // Focal length calculation, yes, just google it if you need

    // ! PROJECTION MATRIX - Converts 3D to 2D screen coordinates
    let mut proj: [[f32; 4]; 4] = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0], 
        [0.0, 0.0, z_far / (z_far - z_near), 1.0],
        [0.0, 0.0, -(z_far * z_near) / (z_far - z_near), 0.0],
    ];
    // ! VIEW MATRIX - Camera position (currently looking from [0,0,5])
    let mut view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 8.0, 1.0], // * This third value is camera position, but remeber, in OpenGL and THREE.js we use -Z to go back, in Vulkan we use +Z, so Z=350 in vulkan is Z=-350 in THREE.js/OpenGL
    ];
    let mut eye_pos = [view[3][0],view[3][1],view[3][2]];

    // Initialize Vulkan Base
    let base = renderer::init_vulkan(&event_loop);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(base.device.clone()));
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        base.device.clone(), 
        StandardCommandBufferAllocatorCreateInfo::default()
    );
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(base.device.clone());

    // Setup Swapchain
    let (mut swapchain, images) = renderer::swapchain::create_swapchain_and_images(
        base.device.clone(), 
        base.surface.clone(), 
        &base.window
    );

    // Setup RenderPass and Pipeline (Logic moved to renderer/pipeline.rs usually)
    // For now, assume you have a function or keep the local setup
    let render_pass = vulkano::single_pass_renderpass!(
        base.device.clone(),
        attachments: {
            color: { load: Clear, store: Store, format: swapchain.image_format(), samples: 1 },
            depth: { load: Clear, store: DontCare, format: vulkano::format::Format::D16_UNORM, samples: 1 }
        },
        pass: { color: [color], depth_stencil: {depth} }
    ).unwrap();

    

    let mut framebuffers = renderer::swapchain::create_framebuffers(&images, render_pass.clone(), &memory_allocator);

     let vertex_input_state = VertexInputState::new()
    .binding(0, VertexInputBindingDescription {
        stride: std::mem::size_of::<VertexPosColorNormal>() as u32, // this might be 48 bytes, I guess, bc I have [[f32; 3], 4] in this color
        input_rate: VertexInputRate::Vertex,
    })
    .binding(1, VertexInputBindingDescription {
        stride: std::mem::size_of::<InstanceData>() as u32,
        input_rate: VertexInputRate::Instance { divisor: 1 },
    })
    .attribute(0, VertexInputAttributeDescription {
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 0,
    })
    .attribute(1, VertexInputAttributeDescription {
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 12,
    })
    .attribute(2,VertexInputAttributeDescription {
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 24,
    })
    .attribute(3, VertexInputAttributeDescription {
        binding: 0, format: Format::R32G32B32_SFLOAT, offset: 36,
    })
    .attribute(4,VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 0,
    })
    .attribute(5,VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 16,
    })
    .attribute(6,VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 32,
    })
    .attribute(7, VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 48, 
    })
    .attribute(8, VertexInputAttributeDescription {
        binding: 1, 
        format: Format::R32G32B32_SFLOAT,
        offset: 64, 
    })
    .attribute(9, VertexInputAttributeDescription {
        binding: 1, 
        format: Format::R32G32B32A32_SFLOAT, 
        offset: 80,
    });


    let library = VulkanLibrary::new().expect("No Vulkan driver found.");
    let required_extensions = vulkano_win::required_extensions(&library);

     let instance = OtherInstance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).unwrap();
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
            PhysicalDeviceType::DiscreteGpu => 0,  // Prefer discrete GPU like RTX...
            PhysicalDeviceType::IntegratedGpu => 1, // Then integrated like a proccessor if it can display Graphic
            _ => 2,                                   // Anything else last
        }).unwrap();


    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
        ..Default::default()
    }).unwrap();
    let queue = queues.next().unwrap(); 
    let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default());

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();
    // ! GRAPHICS PIPELINE - The complete configuration for drawing
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(vertex_input_state) // This is GPU settings
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .rasterization_state(RasterizationState::new().cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::None)) 
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let mut mouse_state = MouseState::default();
    let mut prev_mouse_state = MouseState::default();

    let mut scene = RenderScene::new(&memory_allocator, &descriptor_set_allocator, &pipeline, 3, 10000);
    scene.add_instance(
        create_sphere_subdivided(&memory_allocator, [1.0,1.0,1.0], 2), 
        Instance {
            transform: Transform { 
                position: [-4.0, 0.0, 0.0], 
                ..Default::default() 
            },
            color: [0.95, 0.6, 0.35],        
            shininess: 400.0,                 
            specular_strength: 1.0,          
            roughness: 0.05,                  
            metalness: 1.0,                  
            ..Default::default()
        }
    );

    let mut camera = Camera {
        position: [0.0, 0.0, 10.0],
        yaw: -90.0f32.to_radians(),
        pitch: 0.0,
    };

    let mut previous_frame_end = Some(sync::now(base.device.clone()).boxed());
    let mut recreate_swapchain = false;
    let mut frame_index = 3;
    let start_time = std::time::Instant::now(); 
    let mut dims: [u32; 2] = window.inner_size().into();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
            
            Event::MainEventsCleared => {
                let elapsed = start_time.elapsed().as_secs_f32();
                frame_index = (frame_index + 1) % 3;
                // 1. Handle Swapchain recreation
                if recreate_swapchain {
                    let (new_sw, new_img) = swapchain.recreate(SwapchainCreateInfo {
                        image_extent: base.window.inner_size().into(),
                        ..swapchain.create_info()
                    }).unwrap();
                    swapchain = new_sw;
                    framebuffers = renderer::swapchain::create_framebuffers(&new_img, render_pass.clone(), &memory_allocator);
                    recreate_swapchain = false;
                }

                let (img_index, suboptimal, acquire_future) = 
                    match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => { recreate_swapchain = true; return; }
                        Err(e) => panic!("{e}"),
                    };

                if suboptimal { recreate_swapchain = true; }

                // ! BUILD COMMAND BUFFER - Record all drawing commands
                let mut builder = AutoCommandBufferBuilder::primary(
                    &cb_allocator, 
                    queue.queue_family_index(), 
                    CommandBufferUsage::OneTimeSubmit
                ).unwrap();
                // if mouse_state.inside_window {
                //     println!("Mouse: {:?}", mouse_state.position); // * uncomment if you want to check/understand mouse movement 
                // }
                // * Start render pass (clearing color to dark and depth to 1.0)
                builder.begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()),  // Clear color. You can set to something other, its just a 'background color' 
                            Some(1.0.into()),                    // Clear depth
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffers[img_index as usize].clone())
                    },
                    SubpassContents::Inline,
                ).unwrap()
                .set_viewport(0, vec![Viewport {
                    origin: [0.0, 0.0], // ! can animate camera with that shit
                    dimensions: [dims[0] as f32, dims[1] as f32],
                    depth_range: 0.0..1.0,
                }])
                .bind_pipeline_graphics(pipeline.clone());

                // Scene render, entry point for my future library
                // When I will release 1.0 version of library, I will try to avoid API changes
                scene.update(elapsed, &mouse_state);
                scene.render(&mut builder, &pipeline, &memory_allocator, frame_index, view, proj, eye_pos);

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