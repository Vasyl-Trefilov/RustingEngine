use vulkano::pipeline::graphics::vertex_input::Vertex;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferObject {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub eye_pos: [f32; 3]
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
                [0.0, 0.0, 5.0, 1.0],
            ],
            proj: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (z_far + z_near) / (z_far - z_near), 1.0],
                [0.0, 0.0, -(2.0 * z_far * z_near) / (z_far - z_near), 0.0],
            ],
            eye_pos: [
                0.0, 0.0, 5.0
            ]
        }
    }
}

