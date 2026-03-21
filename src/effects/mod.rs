use crate::scene::InstanceHandle;
use crate::shapes::Mesh;
use crate::RenderScene;
use crate::AnimationType;
use std::sync::Arc;
use crate::Instance;
use crate::Transform;
use rand::*;

pub struct FountainSettings {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub gravity: f32,
}
impl Default for FountainSettings {
    fn default() -> Self {
        Self { position: [0.0, 0.0, 0.0], color: [0.0, 0.5, 1.0], gravity: -0.008 }
    }
}

pub struct FireSettings {
    pub position: [f32; 3],
    pub max_height: f32,
    pub spread: f32,
}
impl Default for FireSettings {
    fn default() -> Self {
        Self { position: [0.0, 0.0, 0.0], max_height: 5.0, spread: 1.0 }
    }
}

pub struct SphereSettings {
    pub center: [f32; 3],
    pub radius: f32,
    pub rotation_speed: f32,
    pub random_color: bool
}
impl Default for SphereSettings {
    fn default() -> Self {
        Self { center: [0.0, 0.0, 0.0], radius: 100.0, rotation_speed: 0.1, random_color: false }
    }
}

pub struct RainSettings {
    pub area: [f32; 3], // Width, Height, Depth
    pub speed: f32,
}
impl Default for RainSettings {
    fn default() -> Self {
        Self { area: [40.0, 40.0, 40.0], speed: 0.3 }
    }
}


pub fn create_fountain(scene: &mut RenderScene, triangle: Mesh, count: u32, settings: Option<FountainSettings>) -> Vec<InstanceHandle> {
    let s = settings.unwrap_or_default();
    let mut rng = rand::rng();
    let grav = s.gravity;
    let base_pos = s.position;

    let fountain_logic = AnimationType::Custom(Arc::new(move |transform, velocity, original_pos, _color, _elapsed| {
        velocity[1] += grav; 
        
        transform.position[0] += velocity[0];
        transform.position[1] += velocity[1];
        transform.position[2] += velocity[2];
        
        transform.rotation[0] += velocity[0] * 5.0;
        transform.rotation[1] += velocity[1] * 2.0;
        
        if transform.position[1] < base_pos[1] - 2.0 {
            transform.position = original_pos.clone();
            velocity[1] = 0.15 + (rand::random::<f32>() * 0.1); 
        }
        
        let life = ((transform.position[1] - base_pos[1]) + 2.0) / 10.0;
        let scale = (0.1 + life).clamp(0.05, 0.3);
        transform.scale = [scale; 3];
    }));

    let mut handles = Vec::new();
    for i in 0..count {
        let angle = (i as f32) * (std::f32::consts::PI * 2.0 / count as f32);
        let speed = rng.random_range(0.02..0.05);
        
        let handle = scene.add_instance(triangle.clone(), Instance {
            transform: Transform { position: base_pos, scale: [0.1; 3], ..Default::default() },
            original_position: base_pos,
            animation: fountain_logic.clone(),
            velocity: [angle.cos() * speed, 0.1 + rng.random_range(0.0..0.1), angle.sin() * speed],
            color: s.color,
            metalness: 0.8, 
            ..Default::default()
        });
        handles.push(handle);
    }
    handles
}

pub fn create_fire(scene: &mut RenderScene, mesh: Mesh, count: u32, settings: Option<FireSettings>) -> Vec<InstanceHandle> {
    let s = settings.unwrap_or_default();
    let mut rng = rand::rng();
    let base_pos = s.position;
    let max_h = s.max_height;

    let fire_logic = AnimationType::Custom(Arc::new(move |transform, velocity, original_pos, _instance_color, _elapsed| {
        let height = transform.position[1] - original_pos[1];
        let sway = (height * 5.0).sin() * 0.07;
        
        transform.position[0] += velocity[0] + sway;
        transform.position[1] += velocity[1];
        transform.position[2] += velocity[2] + (height * 3.0).cos() * 0.07;

        let life = (height / max_h).clamp(0.0, 1.0);
        transform.scale = [0.3 * (1.0 - life); 3];
        transform.rotation[0] += 0.3;

        if height > max_h || (rand::random::<f32>() > 0.98) {
            transform.position = original_pos.clone(); 
            transform.position[0] += (rand::random::<f32>() - 0.5) * s.spread;
            transform.position[2] += (rand::random::<f32>() - 0.5) * s.spread;
            velocity[1] = 0.06 + (rand::random::<f32>() * 0.04);
        }
    }));

    let mut handles = Vec::new();

    for _ in 0..count {
        let color = [1.0, rng.random_range(0.1..0.4), 0.0];
        let handle = scene.add_instance(mesh.clone(), Instance {
            transform: Transform { position: [base_pos[0], base_pos[1] + rng.random_range(0.0..max_h), base_pos[2]], ..Default::default() },
            original_position: base_pos,
            animation: fire_logic.clone(),
            velocity: [0.0, rng.random_range(0.05..0.1), 0.0],
            color, metalness: 0.8,
            ..Default::default()
        });
        handles.push(handle);
    }
    handles
}

pub fn create_void_fire(scene: &mut RenderScene, mesh: Mesh, count: u32, settings: Option<FireSettings>) -> Vec<InstanceHandle> {
    let s = settings.unwrap_or_default();
    let mut rng = rand::rng();
    let base_pos = s.position;

    let fire_logic = AnimationType::Custom(Arc::new(move |transform, velocity, original_pos, _color, elapsed| {
        let height = transform.position[1] - original_pos[1];
        let swirl = 5.0;
        transform.position[0] = original_pos[0] + (elapsed * swirl + height).sin() * (height * 0.2);
        transform.position[2] = original_pos[2] + (elapsed * swirl + height).cos() * (height * 0.2);
        transform.position[1] += velocity[1];

        let life = (height / s.max_height).clamp(0.0, 1.0);
        let p = 0.4 * (1.0 - life) * (1.0 + (elapsed * 10.0).sin() * 0.2);
        transform.scale = [p; 3];

        if height > s.max_height {
            transform.position = original_pos.clone();
            transform.position[1] += (rand::random::<f32>() - 0.5) * 0.5;
        }
    }));
    let mut handles = Vec::new();
    for _ in 0..count {
        let handle = scene.add_instance(mesh.clone(), Instance {
            transform: Transform { position: base_pos, ..Default::default() },
            original_position: base_pos,
            animation: fire_logic.clone(),
            velocity: [0.0, rng.random_range(0.04..0.1), 0.0],
            color: [1.0, 1.0, 1.0],
            metalness: 1.0, 
            ..Default::default()
        });
        handles.push(handle);
    }
    handles
}

pub fn create_event_horizon(scene: &mut RenderScene, mesh: Mesh, count: u32, settings: Option<SphereSettings>) -> Vec<InstanceHandle> {
    let s = settings.unwrap_or_default();
    let mut rng = rand::rng();
    let center = s.center;

    let vortex_logic = AnimationType::Custom(Arc::new(move |transform, _velocity, _orig, _color, _elapsed| {
        let rel_x = transform.position[0] - center[0];
        let rel_z = transform.position[2] - center[2];
        let dist = (rel_x.powi(2) + rel_z.powi(2)).sqrt();
        
        let speed = 2.5 / (dist + 0.1);
        let pull = 0.4;
        
        let x = rel_x;
        let z = rel_z;
        transform.position[0] = center[0] + (x * (speed * 0.02).cos() - z * (speed * 0.02).sin()) - (rel_x / dist) * pull;
        transform.position[2] = center[2] + (x * (speed * 0.02).sin() + z * (speed * 0.02).cos()) - (rel_z / dist) * pull;

        transform.rotation[0] += speed * 0.1;
        transform.rotation[1] += speed * 0.1;
        transform.rotation[2] += speed * 0.05;

        if dist < 1.5 {
            let angle = rand::random::<f32>() * 6.28;
            let spawn_dist = s.radius;
            transform.position = [
                center[0] + angle.cos() * spawn_dist, 
                center[1] + (rand::random::<f32>() - 0.5) * 5.0, 
                center[2] + angle.sin() * spawn_dist
            ];
        }

        let s_val = (dist * 0.01).clamp(0.02, 0.4);
        transform.scale = [s_val, s_val * 2.0, s_val]; 
    }));
    let mut handles = Vec::new();
    for _ in 0..count {
        let angle: f32 = rng.random_range(0.0..6.28);
        let d = rng.random_range(5.0..s.radius);
        let color;
        if s.random_color == false {
            color = [1.0, 1.0, 1.0];
        } else {
            // color = [rng.random_range(0.0..1.0), rng.random_range(0.0..1.0), rng.random_range(0.0..1.0)];
            color = [rng.random_range(0.5..1.0), rng.random_range(0.5..1.0), 0.0];
        }
        
        let handle = scene.add_instance(mesh.clone(), Instance {
            transform: Transform {
                position: [center[0] + angle.cos() * d, center[1] + rng.random_range(-2.0..2.0), center[2] + angle.sin() * d],
                rotation: [rng.random_range(0.0..6.28), rng.random_range(0.0..6.28), 0.0],
                ..Default::default()
            },
            animation: vortex_logic.clone(),
            color: color,
            metalness: 1.0,        
            roughness: 0.05,        
            ..Default::default()
        });
        handles.push(handle);
    }
    handles
}

pub fn create_monochrome_rain(scene: &mut RenderScene, mesh: Mesh, count: u32, settings: Option<RainSettings>) -> Vec<InstanceHandle> {
    let s = settings.unwrap_or_default();
    let mut rng = rand::rng();
    let half_w = s.area[0] / 2.0;
    let height = s.area[1];
    let half_d = s.area[2] / 2.0;

    let rain_logic = AnimationType::Custom(Arc::new(move |transform, velocity, _orig, _color, _elapsed| {
        transform.position[1] -= velocity[1];
        if transform.position[1] < -height/2.0 { transform.position[1] = height/2.0; }
        transform.scale = [0.02, 0.6, 0.02];
    }));
    let mut handles = Vec::new();
    for _ in 0..count {
        let pos = [rng.random_range(-half_w..half_w), rng.random_range(-height/2.0..height/2.0), rng.random_range(-half_d..half_d)];
        let handle = scene.add_instance(mesh.clone(), Instance {
            transform: Transform { position: pos, ..Default::default() },
            animation: rain_logic.clone(),
            velocity: [0.0, rng.random_range(s.speed..s.speed * 2.0), 0.0],
            color: [0.7, 0.8, 1.0], roughness: 1.0, ..Default::default()
        });
    handles.push(handle);
    }
    handles
}

pub fn create_nebula_sphere(scene: &mut RenderScene, mesh: Mesh, count: u32, settings: Option<SphereSettings>) -> Vec<InstanceHandle> {
    let s = settings.unwrap_or_default();
    let mut rng = rand::rng();
    let center = s.center;

    let stars_logic = AnimationType::Custom(Arc::new(move |transform, velocity, original_pos, _color, elapsed| {
        let angle = elapsed * velocity[0];
        let rel_x = original_pos[0] - center[0];
        let rel_z = original_pos[2] - center[2];
        
        transform.position[0] = center[0] + rel_x * angle.cos() - rel_z * angle.sin();
        transform.position[2] = center[2] + rel_x * angle.sin() + rel_z * angle.cos();
        
        let pulse = 0.1 + (elapsed * velocity[1] + original_pos[0]).sin().abs() * 0.15;
        transform.scale = [pulse; 3];
    }));
    let mut handles = Vec::new();
    for _ in 0..count {
        let radius = rng.random_range(s.radius * 0.8..s.radius);
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let phi = rng.random_range(0.0..std::f32::consts::PI);
        let x = center[0] + radius * phi.sin() * theta.cos();
        let y = center[1] + radius * phi.sin() * theta.sin();
        let z = center[2] + radius * phi.cos();

        let handle = scene.add_instance(mesh.clone(), Instance {
            transform: Transform { position: [x, y, z], ..Default::default() },
            original_position: [x, y, z],
            animation: stars_logic.clone(),
            velocity: [rng.random_range(0.05..s.rotation_speed), rng.random_range(1.0..3.0), 0.0],
            color: [1.0, 1.0, 1.0], roughness: 1.0, ..Default::default()
        });
     handles.push(handle);
    }
    handles
}