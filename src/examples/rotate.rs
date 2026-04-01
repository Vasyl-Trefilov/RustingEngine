use rusting_engine::{ComputeShaderType, Engine, Material, Physics, ShaderType, Transform, CollisionType};
/// # Rotating test
/// This example spawns a sphere on the on the floor and cube that fall on it and roll
/// This test shows how accurate and 'good' cs_max( FullPhysic ) is
pub fn main() {
    let mut engine = Engine::new("RustingEngine - Rotation test");
    engine.set_light([30.0, 50.0, 30.0], [1.0, 1.0, 1.0], 500.0);

    // Cube is falling on sphere and rolling from it
    engine.add_cube(
        Transform {
            position: [0.5, 20.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            ..Default::default()
        },
        &Material::standard()
            .color([0.5, 0.5, 1.0])
            .shader(ShaderType::Pbr)
            .build(),
        &Physics::default()
            .gravity(0.2)
            .mass(1.0)
            .velocity([0.0, 0.0, 0.0, 0.5])  // 0.5 is collision radius
            .collision(CollisionType::Box), // type, if < 0.5 => Box, > 0.5 => Sphere
        );

    engine.add_sphere(
        Transform {
            position: [0.0, 10.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            ..Default::default()
        },
    &Material::standard()
            .color([1.0, 0.1, 0.1])
            .shader(ShaderType::Pbr)
            .build(),
    &Physics::default()
            .compute_shader(ComputeShaderType::FullPhysics)
            .mass(1.0)
            .velocity([0.0, 0.0, 0.0, 1.0])  // 10 is collision radius
            .collision(CollisionType::Sphere), // type, if < 0.5 => Box, > 0.5 => Sphere
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
            .gravity(0.0)
            .mass(1000000.0)
            .velocity([0.0, 0.0, 0.0, 20.0])  // 20 is collision radius
            .collision(CollisionType::Box), // type, if < 0.5 => Box, > 0.5 => Sphere
        );
    engine.run();
}