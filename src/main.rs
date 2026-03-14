// ! MAIN ENTRY POINT - This file contains the core Vulkan rendering engine
// * Vulkan is a low-level graphics API that gives us full control over the GPU

// Import all the Vulkan functionality we need
use vulkano::pipeline::Pipeline;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance as OtherInstance, InstanceCreateInfo};
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
use std::io::pipe;
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
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use std::panic;
use rand::prelude::*;
mod scene_manager;
use scene_manager::RenderScene;
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::pipeline::graphics::vertex_input::{
    VertexInputBindingDescription, 
    VertexInputRate
};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;


mod shapes;
use shapes::{VertexPosColor, Mesh, Scene, SceneObject, Transform};
use shapes::shapes::{
    create_cube, create_sphere, create_plane,
    create_cone, create_cylinder, create_dodecahedron, 
    create_grid, create_icosahedron, create_octahedron, 
    create_pyramid, create_sphere_subdivided, create_tetrahedron,
    create_torus, // ! thats looks so sick, I am proud of myself, just a bit
};

use crate::shapes::shapes::create_triangle;

// ! MOUSE 
#[derive(Clone, Copy, Debug)]
struct MouseState {
    position: (f32, f32),      // Normalized coordinates (-1 to 1), so its like a vulkan type
    pixel_position: (f32, f32), // Pixel coordinates, like from 0 to 1900
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
// * very soon I will delete it and use Instance for performance 
#[derive(Clone)]
struct RenderObject {
    per_frame_data: Vec<(Subbuffer<UniformBufferObject>, Arc<PersistentDescriptorSet>)>,
    mesh: Mesh,                                     // The actual geometry data
    transform: Transform,                           // Position/rotation/scale
    animation_type: AnimationType,                  // ? Type of animation (e.g., "Rotate", "Pulse"), I am so unsure about this, I want to do something like in THREE.js 
    mouse_state: MouseState,
    original_position: [f32; 3], 
}

#[derive(Clone)]
pub struct Instance {
    pub transform: Transform,
    pub original_position: [f32; 3],
    pub animation: AnimationType,
    pub velocity: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],  // 4x4 matrix = 16 floats, important for next steps
}

pub struct RenderBatch {
    pub mesh: Mesh,
    pub instances: Vec<Instance>,
}

// * So, this might be like an new version of AnimationType
trait Behavior {
    fn update(&mut self, transform: &mut Transform, time: f32, mouse: &MouseState);
}

struct RotateBehavior;

impl Behavior for RotateBehavior {
    fn update(&mut self, transform: &mut Transform, time: f32, mouse: &MouseState) {
        transform.rotation[1] = -mouse.position.0 * std::f32::consts::PI;
        transform.rotation[0] = -mouse.position.1 * std::f32::consts::PI;
    }
}

// ? I dont want to describe it, maybe I will delete it, bc I dont like it.
#[derive(Clone)]
pub enum AnimationType {
    Rotate,
    Pulse,
    Static,
    Custom(Arc<dyn Fn(&mut Transform, &mut [f32; 3], &mut [f32; 3], f32) + Send + Sync>),
}

impl std::fmt::Debug for AnimationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnimationType::Rotate => write!(f, "Rotate"),
            AnimationType::Pulse => write!(f, "Pulse"),
            AnimationType::Static => write!(f, "Static"),
            AnimationType::Custom(_) => write!(f, "Custom Logic"),
        }
    }
}

// impl Clone for AnimationType {
//     fn clone(&self) -> Self {
//         match self {
//             AnimationType::Rotate => AnimationType::Rotate,
//             AnimationType::Pulse => AnimationType::Pulse,
//             AnimationType::Static => AnimationType::Static,
//             AnimationType::Custom(_) => AnimationType::Custom(_),
//         }
//     }
// }


// ! UNIFORM BUFFER OBJECT - Matrix data sent to GPU each frame
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

impl Default for UniformBufferObject {
    fn default() -> Self {
        let aspect = 16.0 / 9.0;
        let fov = 45.0f32.to_radians();
        let f = 1.0 / (fov / 2.0).tan();
        let z_near = 0.1;
        let z_far = 500.0;
        
        Self {
            view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -350.0, 1.0],
            ],
            proj: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (z_far + z_near) / (z_far - z_near), 1.0],
                [0.0, 0.0, -(2.0 * z_far * z_near) / (z_far - z_near), 0.0],
            ],
        }
    }
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
    proj: [[f32; 4]; 4],
    animation_type: AnimationType
) -> RenderObject {
    let mut per_frame_data = Vec::new();
    for _ in 0..3 {
        let uniform_buffer = Buffer::from_data(
            memory_allocator,
            BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() },
            AllocationCreateInfo { usage: MemoryUsage::Upload, ..Default::default() },
            UniformBufferObject { view, proj },  // No model here
        ).unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
        ).unwrap();
        
        per_frame_data.push((uniform_buffer, descriptor_set));
    }

    RenderObject { 
        mesh, 
        transform, 
        per_frame_data, 
        animation_type, 
        mouse_state: MouseState::default(), 
        original_position: transform.position 
    }
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

        // Instance attributes - model matrix as 4 vec4s, as I said, remebmer the 16 in Instance struct
        layout(location = 3) in vec4 model_row0;
        layout(location = 4) in vec4 model_row1;
        layout(location = 5) in vec4 model_row2;
        layout(location = 6) in vec4 model_row3;

        layout(location = 0) out vec3 v_color;
        layout(location = 1) out vec3 v_barycentric;

        layout(set = 0, binding = 0) uniform UniformBufferObject {
            mat4 view;
            mat4 proj;
        } ubo;

        void main() {
            mat4 instance_model = mat4(model_row0, model_row1, model_row2, model_row3);
            
            gl_Position = ubo.proj * ubo.view * instance_model * vec4(position, 1.0);
            
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
                // float min_dist = min(v_barycentric.x, min(v_barycentric.y, v_barycentric.z));
                // if (min_dist < 0.01) {
                    // f_color = vec4(1.0, 1.0, 1.0, 1.0);
                // } else {
                    f_color = vec4(1.0,1.0,1.0,1.0);
                    // discard; // discard for wireframe, and color for collor, if you dont want to see any wireframe at all, you can comment upper lines, bc they build the wireframe
                    // f_color = vec4(v_color,1.0);
                // }
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
    // * I think, I need this comments more then anyone who will read it(no one cares about my code and will never read it)
    let dims = [1920, 1080]; // Placeholder dimensions for projection matrix
    let aspect = dims[0] as f32 / dims[1] as f32;
    let fov = 45.0f32.to_radians();  // Field of view in radians, its like a minecraft fov, if you know
    let z_near = 0.1;                 // Near clipping plane, it means, if some object is 0.1 from camera, it will not be shown
    let z_far = 500.0;                 // Far clipping plane, how far can 'camera' see, you can set like 1000 if you are not developing some AAA game, but if you do, I guess you know better then me what to do
    let f = 1.0 / (fov / 2.0).tan();   // Focal length calculation, yes, just google it if you need

    // ! PROJECTION MATRIX - Converts 3D to 2D screen coordinates
    let proj: [[f32; 4]; 4] = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0], 
        [0.0, 0.0, z_far / (z_far - z_near), 1.0],
        [0.0, 0.0, -(z_far * z_near) / (z_far - z_near), 0.0],
    ];
    // ! VIEW MATRIX - Camera position (currently looking from [0,0,5])
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 350.0, 1.0], // * This third value is camera position, but remeber, in OpenGL and THREE.js we use -Z to go back, in Vulkan we use +Z, so Z=350 in vulkan is Z=-350 in THREE.js/OpenGL
    ];
    // ! So here is formule of reaching a border of the window 'clip = proj * view * model * vec4(position,1)', I would describe with example, but its too big, google if you want

    // ! VULKAN INITIALIZATION - Setting up the connection to the GPU
    let library = VulkanLibrary::new().expect("No Vulkan driver found.");
    let required_extensions = vulkano_win::required_extensions(&library);

    // * Create Vulkan instance (represents the Vulkan library state)
    let instance = OtherInstance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).unwrap();
    // print!("{:?}", instance); // Some useful data, different OS, drivers and etc. => different output

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
            PhysicalDeviceType::DiscreteGpu => 0,  // Prefer discrete GPU like RTX...
            PhysicalDeviceType::IntegratedGpu => 1, // Then integrated like a proccessor if it can display Graphic
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

    // * So this is the hell part, this is some kind of GPU instructions, I use them to tell GPU, where and how does data looks like, like here is 36 bytes per vertex and 64 per Instance, why? Look at Instance struct.
    let vertex_input_state = VertexInputState::new()
    .binding(0, VertexInputBindingDescription {
        stride: std::mem::size_of::<VertexPosColor>() as u32,
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
    .attribute(3,VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 0,
    })
    .attribute(4,VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 16,
    })
    .attribute(5,VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 32,
    })
    .attribute(6, VertexInputAttributeDescription {
        binding: 1,
        format: Format::R32G32B32A32_SFLOAT,
        offset: 48, 
    });

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

    // * Allocator for Command Buffers (the lists of tasks we send to the GPU)
    let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default());
    let mut recreate_swapchain = false;  // Flag for when window resizes
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());  // Track when GPU finishes to avoid vulkan crush
    
    // * Create descriptor set allocator for managing shader resource bindings
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    // * I moved mouse here, so I can animate object, I hope it will work. And it worked, nice :>
    let mut mouse_state = MouseState::default();
    let mut prev_mouse_state = MouseState::default();

    // ! CREATE SCENE OBJECTS 
    let mut render_objects: Vec<RenderObject> = Vec::new();

    // ! Wow, thats look so nice, I am far from THREE.js, but already good

    // ? Do I need to say what this is creating?
    // * yes, I do

    // * Create a cube
    let cube_mesh = create_cube(&memory_allocator, [0.0, 1.0, 0.0]);  

    // * Create a sphere
    // let sphere_mesh = create_sphere(&memory_allocator, [1.0, 0.0, 0.0], 8, 4);  

    // * Create a subdivided sphere
    let sphere_sub_mesh = create_sphere_subdivided(&memory_allocator, [1.0, 0.0, 0.0], 0);  

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

    // let meshes = [cube_mesh, sphere_mesh, sphere_sub_mesh, plane_mesh, tetra_mesh, octa_mesh, ico_mesh, dodeca_mesh, grid_mesh,torus_mesh, cylinder_mesh, cone_mesh, pyramid_mesh];


    // ! A SCENE, now I will start render with scene, its already very solid, I want to make it more easy for people
    let mut scene = RenderScene::new(&memory_allocator, &descriptor_set_allocator, &pipeline, 3, 1000);

    let triangle = create_triangle(&memory_allocator, [1.0,1.0,1.0]);
    let tetra_mesh = create_tetrahedron(&memory_allocator, [1.0, 0.0, 0.0]); 
    let mut rng = rand::rng(); // ! If chatGPT or any other model will tel you to do rand:thread_rng() and then thread_generate() or something like that, dont listen, they have old data
    // * I create a cool animation with random objects, here you can find cool Transform, velocity and etc. usage
    // let meshes = [cube_mesh, sphere_sub_mesh, tetra_mesh, octa_mesh, ico_mesh, torus_mesh, cylinder_mesh, cone_mesh, pyramid_mesh];

    // ! SO THIS IS HUGE, it render a 100k triangels and create a rotating sphere, with my GPU(rtx 3050ti mobile) it render in ~280fps middle, and OpenGL SUCKS, BC IT RENDER ONLY IN 60 FPS with same logic, but on low level, not like my library

    let mut scene = RenderScene::new(&memory_allocator, &descriptor_set_allocator, &pipeline, 3, 100000);

    let triangle = create_triangle(&memory_allocator, [1.0,1.0,1.0]);
    let mut rng = rand::rng();
    let stars_logic = AnimationType::Custom(Arc::new(|transform, _velocity, original_pos, elapsed| {
        let speed = 0.1; 
        let angle = elapsed * speed;
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        transform.position[0] = original_pos[0] * cos_a - original_pos[2] * sin_a;
        transform.position[2] = original_pos[0] * sin_a + original_pos[2] * cos_a;
        
    }));

    for _ in 0..100000 {
        let radius = 100.0; 
        
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let phi = rng.random_range(0.0..std::f32::consts::PI);
        
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

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
            }
        );
    }

    // for _ in 0..30 {
    //     let x = rng.random_range(-7.0..7.0);
    //     let y = rng.random_range(-7.0..7.0);
    //     let mesh_index = rng.random_range(0..meshes.len());
    //     let mut scale = 0.0;
    //     if mesh_index == 4 {
    //         scale = rng.random_range(0.3..0.5);
    //     } else {
    //         scale = rng.random_range(0.5..0.8);
    //     }

    //     let angle = rng.random_range(0.0..std::f32::consts::TAU);
    //     let speed = 0.005; 
    //     let vx = angle.cos() * speed;
    //     let vy = angle.sin() * speed;

    //     scene.add_instance(
    //         // tetra_mesh.clone(),
    //         meshes[mesh_index].clone(),
    //         Instance {
    //             transform: Transform {
    //                 position: [x, y, 0.0],
    //                 scale: [scale,scale,scale],
    //                 ..Default::default()
    //             },
    //             original_position: [x, y, 0.0],
    //             animation: AnimationType::Rotate, // I added Animation to rotate object, bc its easy and built in by me, but for more complex animation, I will create a seperate one.
    //             velocity: [vx, vy, 0.0],
    //         }
    //     );
    // }
    // cube.animation_type = AnimationType::Custom(Box::new(|transform, elapsed| {
    //     transform.position[0] = elapsed.cos() * 2.0;
    //     transform.rotation[0] = elapsed;
    // }));
    // cube.animation_type = AnimationType::Pulse; // * you can uncomment this shit if you want 'cool' animations
    // render_objects.push(mesh); // * I dont use it anymore


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
                        // if !is_pressed && prev_mouse_state.left_pressed {
                        //     mouse_state.left_clicked = true;
                        // } else {
                        //     mouse_state.left_clicked = false;
                        // }
                        // mouse_state.left_pressed = is_pressed;
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
                        device.wait_idle().unwrap();  // Wait for GPU to finish or it will crash
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
                scene.render(&mut builder, &pipeline, &memory_allocator, frame_index, view, proj);

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


// ! This is cool animation that push objects away from mouse
fn apply_mouse_repulsion(transform: &mut Transform, _original_pos: [f32; 3], mouse_state: &MouseState) {
    if !mouse_state.inside_window { return; }

    let repulsion_radius = 2.5;
    let repulsion_strength = 0.15;
    
    let aspect = 1920.0 / 1080.0;
    let world_h = 10.0 * (45.0f32.to_radians() / 2.0).tan();
    let world_w = world_h * aspect;

    let mouse_world = [
        mouse_state.position.0 * world_w,
        mouse_state.position.1 * world_h, 
        0.0
    ];
    
    let from_mouse = [
        transform.position[0] - mouse_world[0],
        transform.position[1] - mouse_world[1],
    ];
    
    let dist_sq = from_mouse[0]*from_mouse[0] + from_mouse[1]*from_mouse[1];
    
    if dist_sq < repulsion_radius * repulsion_radius && dist_sq > 0.001 {
        let dist = dist_sq.sqrt();
        let force = (1.0 - dist / repulsion_radius) * repulsion_strength;
        
        transform.position[0] += (from_mouse[0] / dist) * force;
        transform.position[1] += (from_mouse[1] / dist) * force;
    }
}

// ! This is 100% useful, when object hit the window border, it pushes it away
fn constrain_to_screen(transform: &mut Transform, original_pos: [f32; 3]) {
    let aspect = 1920.0 / 1080.0;
    let fov_rad = 45.0f32.to_radians();
    
    let distance = 10.0; 
    
    let visible_h = distance * (fov_rad / 2.0).tan();
    let visible_w = visible_h * aspect;

    let margin = 0.9;
    let x_limit = visible_w * margin;
    let y_limit = visible_h * margin;
    
    let mut pushed = false;
    let push_strength = 0.1;

    if transform.position[0] > x_limit {
        transform.position[0] -= push_strength;
        pushed = true;
    } else if transform.position[0] < -x_limit {
        transform.position[0] += push_strength;
        pushed = true;
    }
    
    if transform.position[1] > y_limit {
        transform.position[1] -= push_strength;
        pushed = true;
    } else if transform.position[1] < -y_limit {
        transform.position[1] += push_strength;
        pushed = true;
    }
    
    // ? I dont know when it can be useful
    // if !pushed {
    //     let return_speed = 0.02;
    //     transform.position[0] += (original_pos[0] - transform.position[0]) * return_speed;
    //     transform.position[1] += (original_pos[1] - transform.position[1]) * return_speed;
    //     transform.position[2] += (original_pos[2] - transform.position[2]) * return_speed;
    // }
}