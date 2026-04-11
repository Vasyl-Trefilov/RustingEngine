pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/fragment/pbr.frag"
    }
}

pub mod fs_unlit {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/fragment/unlit.frag"
    }
}

pub mod fs_emissive {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/fragment/emissive.frag"
    }
}

pub mod fs_normal_debug {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/fragment/debug.frag"
    }
}

pub mod fs_heavy {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/fragment/heavy.frag"
    }
}
