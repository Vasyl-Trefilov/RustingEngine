use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let red = materials.add(StandardMaterial {
        base_color: Color::rgb(1.0, 0.2, 0.2),
        unlit: true,
        ..default()
    });

    let green = materials.add(StandardMaterial {
        base_color: Color::rgb(0.2, 1.0, 0.2),
        unlit: true,
        ..default()
    });

    let cube = meshes.add(Cuboid::default());

    let grid_size = 20;

    for x in 0..grid_size {
        for y in 0..25 {
            for z in 0..grid_size {
                let material = if (x + y + z) % 2 == 0 {
                    red.clone()
                } else {
                    green.clone()
                };

                commands.spawn((
                    Mesh3d(cube.clone()),
                    MeshMaterial3d(material),
                    Transform::from_xyz(
                        (x as f32 - grid_size as f32 / 2.0) * 1.5,
                        y as f32 * 1.5 + 5.0,
                        (z as f32 - grid_size as f32 / 2.0) * 1.5,
                    ),
                ));
            }
        }
    }

    // Floor
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(100.0, 1.0, 100.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::rgb(0.4, 0.4, 0.4),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));


	// camera
	commands.spawn((
		Camera3d::default(),
		Transform::from_xyz(60.0, 60.0, 60.0)
			.looking_at(Vec3::ZERO, Vec3::Y),
	));
}