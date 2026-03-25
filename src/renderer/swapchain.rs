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
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    window: &Window,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let caps = device.physical_device().surface_capabilities(surface, Default::default()).unwrap();
    let format = device.physical_device().surface_formats(surface, Default::default()).unwrap()[0].0;
    
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
}

pub fn create_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> std::sync::Arc<RenderPass> {
    vulkano::ordered_passes_renderpass!(
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
        passes: [ {
            color: [color],
            depth_stencil: {depth},
            input: []
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