use std::sync::Arc;
use vulkano::device::Device;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageUsage, SwapchainImage};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, PresentMode};
use vulkano::format::Format;
use vulkano::memory::allocator::StandardMemoryAllocator;
use winit::window::Window;
use vulkano::image::ImageAccess;

pub fn create_swapchain_and_images(
    device: Arc<Device>,
    surface: Arc<Surface>,
    window: &Window,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let caps = device.physical_device().surface_capabilities(&surface, Default::default()).unwrap();
    let format = device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0].0;

    Swapchain::new(
        device,
        surface,
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count,
            image_format: Some(format),
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: caps.supported_composite_alpha.into_iter().next().unwrap(),
            present_mode: PresentMode::Immediate, // High FPS mode
            ..Default::default()
        },
    ).unwrap()
}

pub fn create_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> std::sync::Arc<RenderPass> {
    vulkano::ordered_passes_renderpass!(
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
    ).unwrap()
}

pub fn create_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &StandardMemoryAllocator,
) -> Vec<Arc<Framebuffer>> {
    let dims = images[0].dimensions().width_height();
    
    // Depth buffer is required for 3D sorting
    let depth_image = AttachmentImage::transient(
        memory_allocator,
        dims,
        Format::D16_UNORM,
    ).unwrap();
    let depth_view = ImageView::new_default(depth_image).unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_view.clone()],
                    ..Default::default()
                },
            ).unwrap()
        })
        .collect()
}