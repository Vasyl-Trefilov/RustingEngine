use rusting_engine::{ComputeShaderType, Engine, Material, Physics, ShaderType, Transform};
/// # Stress Test: 10,000 PBR Cubes
/// This example spawns 10,000 objects utilizing standard PBR shading.
/// It highlights frame-rate impact relative to the Unlit version, as PBR 
/// calculates complex lighting, metallic/roughness values, and environment interactions.
/// also good example of usage `set_scene_shader` and `set_scene_physic`
pub fn main() {
    let mut engine = Engine::new("RustingEngine - Stress Test PBR (10k Objects)");
    engine.set_light([30.0, 50.0, 30.0], [1.0, 1.0, 1.0], 500.0);
    let red_pbr = Material::standard()
        .color([1.0, 0.2, 0.2])
        .shader(ShaderType::Pbr)
        .build();
    let green_pbr = Material::standard()
        .color([0.2, 1.0, 0.2])
        .shader(ShaderType::Pbr)
        .build();
    let grid_size = 20; // 20 * 25 * 20 = 10,000 cubes
    
    // Spawn 10,000 interactive physical cubes 
    for x in 0..grid_size {
        for y in 0..25 {
            for z in 0..grid_size {
                let pos = [
                    (x as f32 - grid_size as f32 / 2.0) * 1.5,
                    y as f32 * 1.5 + 7.0,
                    (z as f32 - grid_size as f32 / 2.0) * 1.5,
                ];
                let cube_size = 1.0;
                let radius = cube_size * 0.5;
                // Alternate colors
                let mat = if (x + y + z) % 2 == 0 { &red_pbr } else { &green_pbr };
                engine.add_cube(
                    Transform {
                        position: pos,
                        scale: [cube_size,cube_size,cube_size],
                        ..Default::default()
                    },
                    mat,
                    &Physics::default()
                        .compute_shader(ComputeShaderType::MidPhysic)
                        .velocity([0.0, 0.0, 0.0, radius]) // 20 is collision radius
                        .mass(1.0)
                        .collision(0.2) // type, if < 0.5 => Box, > 0.5 => Sphere
                        .gravity(1.0),
                );
            }
        }
    }

    engine.add_sphere(
        Transform {
            position: [0.0, 200.0, 0.0],
            scale: [20.0, 20.0, 20.0],
            ..Default::default()
        },
    &Material::standard()
            .color([1.0, 0.1, 0.1])
            .shader(ShaderType::Pbr)
            .build(),
    &Physics::default()
            .compute_shader(ComputeShaderType::MidPhysic)
            .mass(1000.0)
            .velocity([0.0, 0.0, 0.0, 40.0])  // 40 is collision radius
            .collision(0.7), // type, if < 0.5 => Box, > 0.5 => Sphere
        );

    // A large floor
    engine.add_cube(
        Transform {
            position: [0.0, 4.0, 0.0],
            scale: [40.0, 1.0, 40.0],
            ..Default::default()
        },
        &Material::standard()
            .color([0.5, 0.5, 0.5])
            .shader(ShaderType::Unlit)
            .build(),
        &Physics::default()
            .compute_shader(ComputeShaderType::MidPhysic)
            .gravity(0.0)
            .mass(1000000000.0)
            .velocity([0.0, 0.0, 0.0, 20.0])  // 20 is collision radius
            .collision(0.2), // type, if < 0.5 => Box, > 0.5 => Sphere
        );

    engine.set_scene_shader(ShaderType::Pbr);
    engine.set_scene_physic(ComputeShaderType::FullPhysics);
    engine.run();
}
