use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::vertex_input::VertexInputBindingDescription;
use crate::{VertexPosColorNormal, shapes::VertexPosColorUv};
use vulkano::pipeline::graphics::vertex_input::VertexInputRate;
use crate::InstanceData;
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::format::Format;
use vulkano::pipeline::GraphicsPipeline;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub view: [[f32; 4]; 4],       // 64 bytes
    pub proj: [[f32; 4]; 4],       // 64 bytes
    pub eye_pos: [f32; 3],         // 12 bytes
    pub padding_1: f32,            // 4 bytes 
    pub light_pos: [f32; 3],       // 12 bytes
    pub padding_2: f32,            // 4 bytes
    pub light_color: [f32; 3],     // 12 bytes
    pub light_intensity: f32,      // 4 bytes
}

impl Default for UniformBufferObject {
    fn default() -> Self {
        Self {
            view: [[0.0; 4]; 4],
            proj: [[0.0; 4]; 4],
            eye_pos: [0.0; 3],
            padding_1: 0.0,
            light_pos: [0.0, 10.0, 0.0],
            padding_2: 0.0,
            light_color: [1.0, 1.0, 1.0],
            light_intensity: 50.0,
        }
    }
}
pub fn create_vertex_input_state() -> VertexInputState {
    VertexInputState::new()
    .binding(0, VertexInputBindingDescription {
        stride: std::mem::size_of::<VertexPosColorUv>() as u32, // this might be 48 bytes, I guess, bc I have [[f32; 3], 4] in this color, not valid shit already
        input_rate: VertexInputRate::Vertex,
    })
    .binding(1, VertexInputBindingDescription {
        stride: std::mem::size_of::<InstanceData>() as u32,
        input_rate: VertexInputRate::Instance { divisor: 1 },
    })
    .attribute(0, VertexInputAttributeDescription {
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 0, // position
    })
    .attribute(1, VertexInputAttributeDescription {
        binding: 0,
        format: Format::R32G32B32_SFLOAT,
        offset: 12, // normal
    })
    .attribute(2, VertexInputAttributeDescription {
        binding: 0,
        format: Format::R32G32_SFLOAT,
        offset: 24, // uv
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
    })
}

use vulkano::shader::ShaderModule;
use std::sync::Arc;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::render_pass::Subpass;
use vulkano::device::Device;
use vulkano::render_pass::RenderPass;

pub fn create_pipeline(vs: Arc<ShaderModule>, fs: Arc<ShaderModule>, render_pass: &Arc<RenderPass>, device: &Arc<Device>) -> std::sync::Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(create_vertex_input_state()) // This is GPU settings
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .rasterization_state(RasterizationState::new().cull_mode(vulkano::pipeline::graphics::rasterization::CullMode::None)) 
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}