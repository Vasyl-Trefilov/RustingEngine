// ! This is test file, here I test how user friendly is my library

use rusting_engine::{Engine, Material, Physics};

fn main() {
    let mut engine = Engine::new("RustingEngine - Simple API Demo");
    engine.set_light([30.0, 30.0, 30.0], [1.0, 0.95, 0.9], 450.0);

    let red = Material::standard().color([1.0, 0.0, 0.0]).build();
    let green = Material::standard().color([0.0, 1.0, 0.0]).build();

    for i in 0..10 {
        for j in 0..3 {
            for k in 0..10 {
                let pos = [i as f32 * 1.5, j as f32 * 2.0 + 7.0, k as f32 * 1.5];
                engine.add_cube(
                    pos,
                    &red,
                    &Physics::default()
                        .velocity([0.0, 0.0, 0.0, 0.5])
                        .mass(1.0)
                        .collision(1.0)
                        .gravity(0.0),
                );
            }
        }
    }

    // engine.add_sphere(
    // [7.0,15.0,7.0],
    //     5.0,
    //     &green,
    //     &Physics::default()
    //             .velocity([0.0, 0.0, 0.0, 5.0])
    //             .mass(100.0)
    //             .collision(1.0)
    //             .gravity(1.0),
    //     );

    engine.add_sphere(
        [7.0, 100.0, 7.0],
        7.5,
        &green,
        &Physics::default()
            .velocity([0.0, 0.0, 0.0, 7.5])
            .mass(100000.0)
            .collision(1.0)
            .gravity(1.0),
    );

    engine.run();
}
