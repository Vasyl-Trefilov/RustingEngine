use std::collections::HashMap;
use std::sync::Arc;

use vulkano::device::Device;
use vulkano::pipeline::ComputePipeline;

use crate::shaders::compute::*;

/// Which descriptor bindings a compute shader needs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShaderBindings {
    pub needs_read_buffer: bool,  // binding 0
    pub needs_write_buffer: bool, // binding 1
    pub needs_grid_counts: bool,  // binding 2
    pub needs_grid_objects: bool, // binding 3
    pub needs_big_indices: bool,  // binding 4
}

impl ShaderBindings {
    pub fn basic() -> Self {
        Self {
            needs_read_buffer: true,
            needs_write_buffer: true,
            needs_grid_counts: false,
            needs_grid_objects: false,
            needs_big_indices: false,
        }
    }

    pub fn grid_build() -> Self {
        Self {
            needs_read_buffer: true,
            needs_write_buffer: true,
            needs_grid_counts: true,
            needs_grid_objects: true,
            needs_big_indices: true,
        }
    }
}

/// Compute shader variant that determines how physics and transform logic is applied.
/// `FullPhysics` is the most powerful one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeShaderType {
    /// Full physics and collision calculations (heavy)
    #[default]
    FullPhysics,
    /// Still very good for performance ( Has collisions, mass and push objects when they collide, but no rotation on push )
    MidPhysic,
    /// Fast copy without physics logic (for static or purely kinematic objects)
    Static,
    /// Fast physics logic (applies velocity and gravity) but skips object collisions check
    NoCollision,
    /// Yes
    GridBuild,
    /// useless shader that has no effect on anything. You can use it to test for fast render without compute shader or I dont know
    Empty,
    /// culling shaders for insane optimization
    Cull,
    /// This is testing not stable shaders
    Test,
}

impl ComputeShaderType {
    pub fn sort_key(&self) -> u32 {
        match self {
            ComputeShaderType::FullPhysics => 0,
            ComputeShaderType::MidPhysic => 1,
            ComputeShaderType::Static => 2,
            ComputeShaderType::NoCollision => 3,
            ComputeShaderType::GridBuild => 4,
            ComputeShaderType::Empty => 5,
            ComputeShaderType::Cull => 6,
            ComputeShaderType::Test => 7,
        }
    }

    pub fn needs_bindings(&self) -> ShaderBindings {
        match self {
            ComputeShaderType::GridBuild => ShaderBindings::grid_build(),
            ComputeShaderType::Test => ShaderBindings::grid_build(),
            _ => ShaderBindings::basic(),
        }
    }
}

pub struct ComputeShaderRegistry {
    pipelines: HashMap<ComputeShaderType, Arc<ComputePipeline>>,
    scene_shader: Option<ComputeShaderType>,
}

impl ComputeShaderRegistry {
    pub fn new(device: &Arc<Device>) -> Self {
        let mut pipelines = HashMap::new();

        // Full physics
        let cs_full =
            cs_full::load(device.clone()).expect("Failed to load FullPhysics compute shader");
        let cp_full = ComputePipeline::new(
            device.clone(),
            cs_full.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create FullPhysics pipeline");
        pipelines.insert(ComputeShaderType::FullPhysics, cp_full);

        // Mid physics
        let cs_mid = cs_no_rot::load(device.clone()).expect("Failed to load MidPhysics compute shader");
        let cp_mid = ComputePipeline::new(
            device.clone(),
            cs_mid.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create FullPhysics pipeline");
        pipelines.insert(ComputeShaderType::MidPhysic, cp_mid);

        let cs_static =
            cs_empty::load(device.clone()).expect("Failed to load Static compute shader");
        let cp_static = ComputePipeline::new(
            device.clone(),
            cs_static.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create Static pipeline");
        pipelines.insert(ComputeShaderType::Static, cp_static);

        // NoCollision (gravity and velocity, but no collision loop)
        let cs_no_col = cs_no_coll::load(device.clone())
            .expect("Failed to load NoCollision compute shader");
        let cp_no_col = ComputePipeline::new(
            device.clone(),
            cs_no_col.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create NoCollision pipeline");
        pipelines.insert(ComputeShaderType::NoCollision, cp_no_col);

        let cs_grid = cs_grid_build::load(device.clone())
            .expect("Failed to load GridBuild compute shader");
        let cp_grid = ComputePipeline::new(
            device.clone(),
            cs_grid.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create GridBuild pipeline");
        pipelines.insert(ComputeShaderType::GridBuild, cp_grid);

        let cs_empty = cs_empty::load(device.clone())
            .expect("Failed to load Empty compute shader");
        let cp_empty = ComputePipeline::new(
            device.clone(),
            cs_empty.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create Empty shader pipeline");
        pipelines.insert(ComputeShaderType::Empty, cp_empty);

        let cs_cull = cs_cull::load(device.clone())
            .expect("Failed to load Cull compute shader");
        let cp_cull = ComputePipeline::new(
            device.clone(),
            cs_cull.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create Cull shader pipeline");
        pipelines.insert(ComputeShaderType::Cull, cp_cull);


        let cs_test = cs_test::load(device.clone()).expect("Failed to load Test compute shader");
        let cp_test = ComputePipeline::new(
            device.clone(),
            cs_test.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create Test pipeline");
        pipelines.insert(ComputeShaderType::Test, cp_test);

        Self {
            pipelines,
            scene_shader: None,
        }
    }

    pub fn get_pipeline(&self, shader_type: ComputeShaderType) -> &Arc<ComputePipeline> {
        self.pipelines
            .get(&shader_type)
            .expect("Compute pipeline not found")
    }

    /// Set a scene-wide shader override. All objects will use this shader,
    /// ignoring their per-object shader setting.
    pub fn set_scene_shader(&mut self, shader: ComputeShaderType) {
        self.scene_shader = Some(shader);
    }

    pub fn get_default_shader(&self) -> ComputeShaderType {
        ComputeShaderType::default()
    }

    /// Clear the scene-wide shader override. Objects will use their per-object shader.
    pub fn clear_scene_shader(&mut self) {
        self.scene_shader = None;
    }

    /// Return the scene physic shader or default
    pub fn scene_shader(&self) -> ComputeShaderType {
        self.scene_shader
            .unwrap_or_else(|| self.get_default_shader())
    }

    /// Return the scene physic shader or None
    pub fn scene_shader_optional(&self) -> Option<ComputeShaderType> {
        self.scene_shader
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullPushConstants {
    pub view_proj: [[f32; 4]; 4],
    pub batch_offset: u32,         // Index of first instance in the physics buffer
    pub batch_count: u32,          // How many instances in this batch
    pub visible_list_offset: u32,  // Where to start writing in the VisibleIndices buffer
}