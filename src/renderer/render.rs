use vulkano::{command_buffer::AutoCommandBufferBuilder};
use std::sync::Arc;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::Framebuffer;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::device::Queue;

// ! BUILD COMMAND BUFFER - Record all drawing commands
pub fn process_render(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, 
    framebuffers: &[Arc<Framebuffer>], 
    img_index: u32, 
    dims: [u32; 2], 
    pipeline: &Arc<GraphicsPipeline>
) {
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![
                    Some([0.0, 0.0, 0.0, 1.0].into()),
                    Some(1.0.into()),
                ],
                ..RenderPassBeginInfo::framebuffer(framebuffers[img_index as usize].clone())
            },
            SubpassContents::Inline,
        )
        .unwrap() 
        .set_viewport(0, vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dims[0] as f32, dims[1] as f32],
            depth_range: 0.0..1.0,
        }])
        .bind_pipeline_graphics(pipeline.clone());
    
    // Notice the semicolon here! 
    // This function just adds commands to the builder created in main.
}

pub fn create_builder(cb_allocator: &StandardCommandBufferAllocator, queue: &Arc<Queue>) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
	AutoCommandBufferBuilder::primary(
		cb_allocator, 
		queue.queue_family_index(), 
		CommandBufferUsage::OneTimeSubmit
	).unwrap()
}