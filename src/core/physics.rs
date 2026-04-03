use crate::core::ComputeShaderType;
use crate::core::collisions::CollisionType;

#[derive(Clone, Debug, Copy)]
pub struct Physics {
    pub compute_shader: ComputeShaderType,
    pub collision_type: CollisionType,
    pub mass: f32,
    pub bounciness: f32,       // 0.0 - rock, 1.0 - super-ball
    pub friction: f32,         // 0.0 - ice, 1.0 - gummy
    pub gravity_scale: f32,    // 1.0 = normal gravity, -1.0 = antigravity
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            compute_shader: ComputeShaderType::FullPhysics,
            collision_type: CollisionType::Box, 
            mass: 1.0,
            bounciness: 0.3, 
            friction: 0.5,
            gravity_scale: 1.0,
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
        }
    }
}

impl Physics {
    pub fn bounciness(mut self, val: f32) -> Self { self.bounciness = val; self }
    pub fn friction(mut self, val: f32) -> Self { self.friction = val; self }
    pub fn gravity_scale(mut self, val: f32) -> Self { self.gravity_scale = val; self }
    pub fn angular_velocity(mut self, val: [f32; 3]) -> Self { self.angular_velocity = val; self }
    pub fn linear_velocity(mut self, v: [f32; 3]) -> Self { self.linear_velocity = v; self }
    pub fn mass(mut self, m: f32) -> Self { self.mass = m; self }
    pub fn collision_type(mut self, c: CollisionType) -> Self { self.collision_type = c; self }
    pub fn compute_shader(mut self, c: ComputeShaderType) -> Self { self.compute_shader = c; self }
}
