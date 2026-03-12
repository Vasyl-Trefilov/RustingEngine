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
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use std::panic;

// Import our custom shape and mesh system
mod shapes;
use shapes::{VertexPosColor, Mesh, Scene, SceneObject, Transform};
use shapes::shapes::{
    create_cube, create_sphere, create_plane,
    create_cone, create_cylinder, create_dodecahedron, 
    create_grid, create_icosahedron, create_octahedron, 
    create_pyramid, create_sphere_subdivided, create_tetrahedron,
    create_torus, // ! thats looks so sick, I am proud of myself, just a bit
};

// ! STAR STRUCTURE - For future particle system (currently unused)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Star {
    position: [f32; 3]  // 3D position of star
}

// ! MOUSE 
#[derive(Clone, Copy, Debug)]
struct MouseState {
    position: (f32, f32),      // Normalized coordinates (-1 to 1)
    pixel_position: (f32, f32), // Pixel coordinates
    left_clicked: bool,
    right_clicked: bool,
    left_pressed: bool,
    right_pressed: bool,
    inside_window: bool,
}

impl Default for MouseState {
    // #[inline] // ? I founded this in library, what does this thing? So I read a bit about this, this is for performance, but its making a compiler time longer, so I will leave it commented, maybe after some time I will uncomment it
    fn default() -> Self {
        Self {
            position: (0.0, 0.0),
            pixel_position: (0.0, 0.0),
            left_clicked: false,
            right_clicked: false,
            left_pressed: false,
            right_pressed: false,
            inside_window: true,
        }
    }
}

// ! PUSH CONSTANTS - Small data we can push directly to shaders (fast!)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    time: f32,      // Current time for animations
    aspect: f32,    // Screen aspect ratio (width/height)
    mouse_pos: [f32; 2],      // Mouse position in normalized coordinates
    mouse_clicked: u32,        // Bit flags: bit0 = left click, bit1 = right click
    mouse_pressed: u32, 
}

// * RenderObject - Represents something we can draw on screen
#[derive(Clone)]
struct RenderObject {
    per_frame_data: Vec<(Subbuffer<UniformBufferObject>, Arc<PersistentDescriptorSet>)>,
    mesh: Mesh,                                     // The actual geometry data
    transform: Transform,                           // Position/rotation/scale
    animation_type: AnimationType,                  // ? Type of animation (e.g., "Rotate", "Pulse"), I am so unsure about this, I want to do something like in THREE.js 
    mouse_state: MouseState,
}


// ? I dont want to describe it, maybe I will delete it, bc I dont like it.
enum AnimationType {
    Rotate,
    Pulse,
    Static,
    Custom(Box<dyn Fn(&mut Transform, f32) + Send + Sync>),
}

impl Clone for AnimationType {
    fn clone(&self) -> Self {
        match self {
            AnimationType::Rotate => AnimationType::Rotate,
            AnimationType::Pulse => AnimationType::Pulse,
            AnimationType::Static => AnimationType::Static,
            AnimationType::Custom(_) => panic!("Cannot clone custom animation"),
        }
    }
}

impl std::fmt::Debug for AnimationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnimationType::Rotate => write!(f, "Rotate"),
            AnimationType::Pulse => write!(f, "Pulse"),
            AnimationType::Static => write!(f, "Static"),
            AnimationType::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
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
    let mut per_frame_data = Vec::new();
    // * Create GPU-side buffer for our uniform data
    for _ in 0..3 {
        let uniform_buffer = Buffer::from_data(
            memory_allocator,
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
        
        per_frame_data.push((uniform_buffer, descriptor_set));
    }

    RenderObject { mesh, transform, per_frame_data, animation_type: AnimationType::Static, mouse_state: MouseState { ..Default::default() }, }
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
        Format::D16_UNORM  
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
                    // discard;
                    f_color = vec4(v_color,0.0);
                }
            }
            
            
        "
    }
}

// ! MAIN FUNCTION - Program entry point
fn main() {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {:?}", panic_info);
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("Panic payload: {}", s);
        }
    }));
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
    // print!("{:?}", instance);
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
            present_mode: PresentMode::Immediate,  // Show frames immediately (no vsync), can be used for benchmarking or just for fun
            // present_mode: PresentMode::Fifo, // So, only Fifo is guaranteed to be supported on every device. And I think its better for some kind of prod, if I ever will get to this point, but for now, I want to see the maximum fps, so I will use Immediate, but if you want to use Fifo, its your choise
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

    // * I moved mouse here, so I can animate object, I hope it will work. And it worked, nice :>
    let mut mouse_state = MouseState::default();
    let mut prev_mouse_state = MouseState::default();
    // ! CREATE SCENE OBJECTS 
    let mut render_objects = Vec::new();

    // ! Wow, thats look so nice, I am far from THREE.js, but already good

    // ? Do I need to say what this is creating?
    // * yes, I do

    // * Create a cube
    let cube_mesh = create_cube(&memory_allocator, [0.0, 1.0, 0.0]);  

    // * Create a sphere
    let sphere_mesh = create_sphere(&memory_allocator, [1.0, 0.0, 0.0], 32, 16);  

    // * Create a subdivided sphere
    let sphere_sub_mesh = create_sphere_subdivided(&memory_allocator, [1.0, 0.0, 0.0], 8);  

    // * Crete a plane, so flat as my female classmates..
    let plane_mesh = create_plane(&memory_allocator, [1.0, 0.0, 0.0], 2.0, 2.0);  

    // * Create a tetrahedron
    let tetra_mesh = create_tetrahedron(&memory_allocator, [1.0, 0.0, 0.0]); 

    // * Create an octahedron
    let octa_mesh = create_octahedron(&memory_allocator, [0.0, 0.0, 1.0]); 

    // * Create an icosahedron
    let ico_mesh = create_icosahedron(&memory_allocator, [1.0, 1.0, 0.0]);  

    // * Create a dodecahedron, why names for that is so sick, I am like back to Biochemistry days, Hexatriacontane
    let dodeca_mesh = create_dodecahedron(&memory_allocator, [1.0, 1.0, 0.0]);  
    
    // * Create a grid, useful for debugging
    let grid_mesh = create_grid(&memory_allocator, [1.0, 1.0, 0.0], 4.0, 8);  

    // * Create a torus
    let torus_mesh = create_torus(&memory_allocator, [1.0, 0.0, 1.0], 1.0, 0.3, 30, 15);  

    // * Create a cylinder
    let cylinder_mesh = create_cylinder(&memory_allocator, [0.0, 1.0, 1.0], 0.5, 1.0, 32); 

    // * Create a cone
    let cone_mesh = create_cone(&memory_allocator, [1.0, 0.5, 0.0], 0.5, 1.0, 32); 

    // * Create a pyramid
    let pyramid_mesh = create_pyramid(&memory_allocator, [0.5, 0.0, 0.5], 1.0, 1.0);

    let meshes = [cube_mesh, sphere_mesh, sphere_sub_mesh, plane_mesh, tetra_mesh, octa_mesh, ico_mesh, dodeca_mesh, grid_mesh,torus_mesh, cylinder_mesh, cone_mesh, pyramid_mesh];

    // You can see how to switch objects in click function 
    let mut mesh = create_render_object(
        &device, &memory_allocator, &descriptor_set_allocator, &pipeline,
        meshes[0].clone(),
        Transform { position: [0.0, 0.0, 0.0], rotation: [0.0, 0.0, 0.0], ..Default::default() }, 
        view, proj
    );
    // cube.animation_type = AnimationType::Custom(Box::new(|transform, elapsed| {
    //     transform.position[0] = elapsed.cos() * 2.0;
    //     transform.rotation[0] = elapsed;
    // }));
    mesh.animation_type = AnimationType::Rotate;
    // cube.animation_type = AnimationType::Pulse; // * you can uncomment this shit if you want 'cool' animations
    render_objects.push(mesh);
    
    // * FPS counter setup
    let start_time = std::time::Instant::now(); 
    let mut frame_count: u32 = 0;
    let mut fps_timer = std::time::Instant::now();
    let mut time: f32 = 0.0;
    let mut frame_index = 0; 
    let mut total_fps = 0;
    let mut mesh_index = 0; // * I use it to view each mesh, so if unused its okay.
    let mut dims: [u32; 2] = window.inner_size().into();
    // if dims[0] == 0 || dims[1] == 0 { return; }  // Window minimized => skip rendering
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
            // ! MOUSE EVENTS, I cant wait
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                let window_size = window.inner_size();
                // Convert to normalized coordinates (-1 to 1)
                let norm_x = (position.x as f32 / window_size.width as f32) * 2.0 - 1.0;
                let norm_y = 1.0 - (position.y as f32 / window_size.height as f32) * 2.0; // Flip Y
                
                mouse_state.position = (norm_x, norm_y);
                mouse_state.pixel_position = (position.x as f32, position.y as f32);
                mouse_state.inside_window = true;
            },
            
            Event::WindowEvent { event: WindowEvent::CursorLeft { .. }, .. } => {
                mouse_state.inside_window = false;
            },
            
            Event::WindowEvent { event: WindowEvent::CursorEntered { .. }, .. } => {
                mouse_state.inside_window = true;
            },
            
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                let is_pressed = match state {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
                
                match button {
                    winit::event::MouseButton::Left => {
                        if !is_pressed && prev_mouse_state.left_pressed {
                            mouse_state.left_clicked = true;
                            
                            mesh_index = (mesh_index + 1) % 13;
                            // Some cool Meshes switch on click
                            let new_mesh = match mesh_index {
                                0 => create_cube(&memory_allocator, [0.0, 1.0, 0.0]),
                                1 => create_sphere(&memory_allocator, [1.0, 0.0, 0.0], 32, 16),
                                2 => create_sphere_subdivided(&memory_allocator, [1.0, 0.0, 0.0], 3), // Lower subdivision!
                                3 => create_plane(&memory_allocator, [1.0, 0.0, 0.0], 2.0, 2.0),
                                4 => create_tetrahedron(&memory_allocator, [1.0, 0.0, 0.0]),
                                5 => create_octahedron(&memory_allocator, [0.0, 0.0, 1.0]),
                                6 => create_icosahedron(&memory_allocator, [1.0, 1.0, 0.0]),
                                7 => create_dodecahedron(&memory_allocator, [1.0, 1.0, 0.0]),
                                8 => create_grid(&memory_allocator, [1.0, 1.0, 0.0], 4.0, 8),
                                9 => create_torus(&memory_allocator, [1.0, 0.0, 1.0], 1.0, 0.3, 20, 10), // Lower segments!
                                10 => create_cylinder(&memory_allocator, [0.0, 1.0, 1.0], 0.5, 1.0, 16), // Lower sectors!
                                11 => create_cone(&memory_allocator, [1.0, 0.5, 0.0], 0.5, 1.0, 16), // Lower sectors!
                                12 => create_pyramid(&memory_allocator, [0.5, 0.0, 0.5], 1.0, 1.0),
                                _ => create_cube(&memory_allocator, [0.0, 1.0, 0.0]),
                            };
                            
                            if let Some(obj) = render_objects.get_mut(0) {
                                obj.mesh = new_mesh;
                            }
                            
                            println!("Switched to mesh {}", mesh_index);
                        } else {
                            mouse_state.left_clicked = false;
                        }
                        mouse_state.left_pressed = is_pressed;
                    },
                    winit::event::MouseButton::Right => {
                        if !is_pressed && prev_mouse_state.right_pressed {
                            mouse_state.right_clicked = true;
                        } else {
                            mouse_state.right_clicked = false;
                        }
                        mouse_state.right_pressed = is_pressed;
                    },
                    _ => {}
                }
            },
            // ! RENDER FRAME - This is where the actual drawing happens
            Event::MainEventsCleared => {
                // unsafe {
                //     device.wait_idle().unwrap(); // So, this shit can fix a lot of problems, BUT ITS ONLY FOR TESTING, PERFORMANCE IS GOING DOWN
                // }

                frame_index = (frame_index + 1) % 3;
                prev_mouse_state = mouse_state;
                // time += 0.002; // this is frame bassed, so I dont like to use it, but if you want, its your choise 
                let elapsed = start_time.elapsed().as_secs_f32();



                // * FPS calculation and display
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

                // ! RECREATE SWAPCHAIN if window was resized
                if recreate_swapchain {
                    unsafe {
                        device.wait_idle().unwrap();  // Wait for GPU to finish
                    }
                    dims = window.inner_size().into();
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
                // if mouse_state.inside_window {
                //     println!("Mouse: {:?}", mouse_state.position); // * uncomment if you want to check/understand mouse movement 
                // }
                // * Start render pass (clearing color to dark and depth to 1.0)
                builder.begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()),  // Clear color. You can set to something other, its just a background 
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
                animate_objects(&mut render_objects, elapsed, &mouse_state, &mut builder, img_index as usize, &pipeline);
                // * Draw all objects

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

fn animate_objects(
    render_objects: &mut Vec<RenderObject>, 
    elapsed: f32, 
    mouse_state: &MouseState,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, 
    frame_index: usize, 
    pipeline: &Arc<GraphicsPipeline>
) {
    for obj in render_objects.iter_mut() {
        obj.mouse_state = *mouse_state;
        // Update (CPU) 
        match &obj.animation_type {
            AnimationType::Rotate => {
                // obj.transform.rotation[1] = elapsed,
                obj.transform.rotation[1] = -mouse_state.position.0 * std::f32::consts::PI;
                obj.transform.rotation[0] = -mouse_state.position.1 * std::f32::consts::PI;
            }
            AnimationType::Pulse => {
                let s = (elapsed.sin() + 1.0) / 2.0;
                obj.transform.scale = [s, s, s];
            },
            AnimationType::Static => {},
            AnimationType::Custom(func) => func(&mut obj.transform, elapsed),
        }

        // Render (GPU)
        builder.bind_vertex_buffers(0, (obj.mesh.vertices.clone(),)); 
    
        let (buffer, descriptor_set) = &obj.per_frame_data[frame_index];
        
        let mut data = buffer.write().unwrap();
        data.model = obj.transform.to_matrix();
        
        builder.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            descriptor_set.clone(), 
        );
        
        if let Some(indices) = &obj.mesh.indices {
            builder.bind_index_buffer(indices.clone());
            builder.draw_indexed(obj.mesh.index_count, 1, 0, 0, 0).unwrap();
        } else {
            builder.draw(obj.mesh.vertex_count, 1, 0, 0).unwrap();
        }
    }
}