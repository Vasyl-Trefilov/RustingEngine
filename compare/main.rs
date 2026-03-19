use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    render::render_asset::RenderAssetUsages
};
use rand::prelude::*;
use std::f32::consts::{PI, TAU};
use std::time::Instant;

// 1. Define a Resource to hold your benchmark data
#[derive(Resource)]
struct BenchmarkStats {
    start_time: Instant,
    frame_count: u32,
    fps_timer: Instant,
    total_frames_accumulated: u32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy 100k Benchmark".into(),
                resolution: (1920.0, 1080.0).into(),
                present_mode: bevy::window::PresentMode::Immediate, // Vsync OFF
                ..default()
            }),
            ..default()
        }))
        // 2. Insert the Resource into the App
        .insert_resource(BenchmarkStats {
            start_time: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            total_frames_accumulated: 0,
        })
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup)
        .add_systems(Update, (update_fps, animate_stars)) // Run both systems
        .run();
}

// 3. Separate FPS logic into its own system for cleanliness
fn update_fps(time: Res<Time>, mut stats: ResMut<BenchmarkStats>) {
    stats.frame_count += 1;
    let elapsed_fps = stats.fps_timer.elapsed().as_secs_f32();
    let total_elapsed = stats.start_time.elapsed().as_secs_f32();

    // Log FPS every 2 seconds (matching your logic)
    if elapsed_fps >= 2.0 {
        let fps = stats.frame_count as f32 / elapsed_fps;
        println!("FPS: {:.0}", fps);
        
        stats.total_frames_accumulated += stats.frame_count;
        stats.frame_count = 0;
        stats.fps_timer = Instant::now();

        // Middle FPS after 10 seconds
        if total_elapsed >= 10.0 {
            let middle_fps = stats.total_frames_accumulated as f32 / total_elapsed;
            println!("middle Fps: {:.0}", middle_fps);
        }
    }
}

fn animate_stars(time: Res<Time>, mut query: Query<(&mut Transform, &Star)>) {
    let elapsed = time.elapsed_secs();
    let speed = 0.1;
    let angle = elapsed * speed;
    let (sin_a, cos_a) = angle.sin_cos();

    // Parallel iterator for maximum performance
    query.par_iter_mut().for_each(|(mut transform, star)| {
        let x0 = star.original_pos.x;
        let z0 = star.original_pos.z;

        transform.translation.x = x0 * cos_a - z0 * sin_a;
        transform.translation.z = x0 * sin_a + z0 * cos_a;
    });
}

#[derive(Component)]
struct Star {
    original_pos: Vec3,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0, 0.1, 0.0], [-0.1, -0.1, 0.0], [0.1, -0.1, 0.0]],
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3]);
    mesh.insert_indices(Indices::U32(vec![0, 1, 2]));

    let mesh_handle = meshes.add(mesh);
    let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    let mut rng = rand::thread_rng();
    let radius = 100.0;

    for _ in 0..100_000 {
        let theta = rng.gen_range(0.0..TAU);
        let phi = rng.gen_range(0.0..PI);
        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material_handle.clone()),
            Transform::from_translation(Vec3::new(x, y, z)).with_scale(Vec3::splat(0.2)),
            Star { original_pos: Vec3::new(x, y, z) },
        ));
    }

    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            range: 500.0,
            color: Color::srgb(1.0, 0.8, 0.6),
            ..default()
        },
        Transform::from_xyz(0.0, 120.0, 0.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 350.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}