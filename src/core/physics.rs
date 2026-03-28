#[derive(Clone, Debug)]
pub struct Physics {
    pub velocity: [f32; 4],
    pub mass: f32,
    pub collision: f32,
    pub gravity: f32,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            velocity: [0.0, 0.0, 0.0, 1.0],
            mass: 1.0,
            collision: 0.0,
            gravity: 1.0,
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

    pub fn collision(mut self, c: f32) -> Self {
        self.collision = c;
        self
    }

    pub fn gravity(mut self, g: f32) -> Self {
        self.gravity = g;
        self
    }
}
