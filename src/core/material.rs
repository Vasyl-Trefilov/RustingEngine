#[derive(Clone, Debug)]
pub struct Material {
    pub color: [f32; 3],
    pub emissive: f32,
    pub roughness: f32,
    pub metalness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            emissive: 0.0,
            roughness: 0.5,
            metalness: 0.0,
        }
    }
}

impl Material {
    pub fn standard() -> MaterialBuilder {
        MaterialBuilder::default()
    }
}

pub struct MaterialBuilder {
    color: [f32; 3],
    emissive: f32,
    roughness: f32,
    metalness: f32,
}

impl Default for MaterialBuilder {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            emissive: 0.0,
            roughness: 0.5,
            metalness: 0.0,
        }
    }
}

impl MaterialBuilder {
    pub fn color(mut self, c: [f32; 3]) -> Self {
        self.color = c;
        self
    }

    pub fn emissive(mut self, e: f32) -> Self {
        self.emissive = e;
        self
    }

    pub fn roughness(mut self, r: f32) -> Self {
        self.roughness = r;
        self
    }

    pub fn metalness(mut self, m: f32) -> Self {
        self.metalness = m;
        self
    }

    pub fn build(self) -> Material {
        Material {
            color: self.color,
            emissive: self.emissive,
            roughness: self.roughness,
            metalness: self.metalness,
        }
    }
}
