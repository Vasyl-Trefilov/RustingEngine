use crate::core::ComputeShaderType;
use crate::core::collisions::CollisionType;

#[derive(Clone, Debug)]
pub struct Physics {
    pub velocity: [f32; 4],
    pub mass: f32,
    pub collision: CollisionType,
    pub gravity: f32,
    pub compute_shader: ComputeShaderType,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            velocity: [0.0, 0.0, 0.0, 1.0],
            mass: 1.0,
            collision: CollisionType::Sphere,
            gravity: 1.0,
            compute_shader: ComputeShaderType::FullPhysics,
        }
    }
}

impl Physics {
    pub fn velocity(mut self, v: [f32; 4]) -> Self {
        self.velocity = v;
        self
    }

    pub fn mass(mut self, m: f32) -> Self {
        self.mass = m;
        self
    }

    pub fn collision(mut self, c: CollisionType) -> Self {
        self.collision = c;
        self
    }

    pub fn gravity(mut self, g: f32) -> Self {
        self.gravity = g;
        self
    }

    pub fn compute_shader(mut self, c: ComputeShaderType) -> Self {
        self.compute_shader = c;
        self
    }
}
