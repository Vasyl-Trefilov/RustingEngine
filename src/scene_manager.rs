// So, here we go, a start of my library, a Scene
use crate::shapes::*;
use crate::{RenderObject, create_render_object, animate_objects};
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use std::sync::Arc;
use crate::MouseState;
use crate::AnimationType;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::Pipeline;

pub struct RenderScene {
    render_objects: Vec<RenderObject>, 
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

impl RenderScene {
    pub fn new(view: [[f32; 4]; 4], proj: [[f32; 4]; 4]) -> Self {
        Self {
            render_objects: Vec::new(),
            view,
            proj,
        }
    }
    
    pub fn add_object(&mut self, object: RenderObject) {
        self.render_objects.push(object);
    }
    
    pub fn update(&mut self, elapsed: f32, mouse_state: &MouseState) {
        for obj in self.render_objects.iter_mut() {
            obj.mouse_state = *mouse_state;
            
            match &obj.animation_type {
                AnimationType::Rotate => {
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
        }
    }
    
    pub fn render(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        frame_index: usize,
        pipeline: &Arc<GraphicsPipeline>,
    ) {
        for obj in self.render_objects.iter_mut() {
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
}