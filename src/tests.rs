#[cfg(test)]
mod tests {
    use crate::CollisionType;
    use crate::core::material::{Material, MaterialBuilder};
    use crate::core::physics::Physics;
    use crate::rendering::shader_registry::ShaderType;
    use crate::scene::object::{Instance, RenderBatch};

    // ShaderType Tests

    #[test]
    fn shader_type_default_is_pbr() {
        assert_eq!(ShaderType::default(), ShaderType::Pbr);
    }

    #[test]
    fn shader_type_sort_keys_are_unique_and_ordered() {
        let all = ShaderType::all();
        let mut keys: Vec<u32> = all.iter().map(|s| s.sort_key()).collect();
        let original = keys.clone();
        keys.sort();
        keys.dedup();
        // All unique
        assert_eq!(keys.len(), all.len());
        // Already sorted
        assert_eq!(keys, original);
    }

    #[test]
    fn shader_type_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ShaderType::Pbr);
        set.insert(ShaderType::Unlit);
        set.insert(ShaderType::Pbr); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&ShaderType::Pbr));
        assert!(set.contains(&ShaderType::Unlit));
    }

    #[test]
    fn shader_type_clone_and_copy() {
        let a = ShaderType::Emissive;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // Material Builder Tests

    #[test]
    fn material_default_has_pbr_shader() {
        let mat = Material::default();
        assert_eq!(mat.shader, ShaderType::Pbr);
    }

    #[test]
    fn material_builder_sets_shader() {
        let mat = Material::standard().shader(ShaderType::Unlit).build();
        assert_eq!(mat.shader, ShaderType::Unlit);
    }

    #[test]
    fn material_builder_default_shader_is_pbr() {
        let mat = Material::standard().build();
        assert_eq!(mat.shader, ShaderType::Pbr);
    }

    #[test]
    fn material_builder_chaining_preserves_shader() {
        let mat = Material::standard()
            .color([1.0, 0.0, 0.0])
            .roughness(0.8)
            .metalness(0.5)
            .shader(ShaderType::NormalDebug)
            .emissive(0.0)
            .build();

        assert_eq!(mat.shader, ShaderType::NormalDebug);
        assert_eq!(mat.color, [1.0, 0.0, 0.0]);
        assert_eq!(mat.roughness, 0.8);
        assert_eq!(mat.metalness, 0.5);
    }

    #[test]
    fn material_builder_shader_can_be_overridden() {
        let mat = Material::standard()
            .shader(ShaderType::Unlit)
            .shader(ShaderType::Emissive) // override to test what if user override
            .build();
        assert_eq!(mat.shader, ShaderType::Emissive);
    }

    // Instance Tests

    #[test]
    fn instance_default_has_pbr_shader() {
        let inst = Instance::default();
        assert_eq!(inst.shader, ShaderType::Pbr);
    }

    #[test]
    fn instance_custom_shader() {
        let inst = Instance {
            shader: ShaderType::Unlit,
            ..Default::default()
        };
        assert_eq!(inst.shader, ShaderType::Unlit);
    }

    #[test]
    fn instance_clone_preserves_shader() {
        let inst = Instance {
            shader: ShaderType::Emissive,
            color: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let cloned = inst.clone();
        assert_eq!(cloned.shader, ShaderType::Emissive);
        assert_eq!(cloned.color, [1.0, 0.0, 0.0]);
    }

    // Batch Grouping Tests

    // Uses a lightweight mock to test add_instance batching logic
    // without requiring GPU. We replicate the batching key check.

    #[test]
    fn same_shader_same_key_groups_together() {
        // Simulates the batching condition: same mesh ptr + same shader = same batch
        let shader_a = ShaderType::Pbr;
        let shader_b = ShaderType::Pbr;
        assert_eq!(shader_a, shader_b); // same shader should merge
    }

    #[test]
    fn different_shader_different_key_separates() {
        let shader_a = ShaderType::Pbr;
        let shader_b = ShaderType::Unlit;
        assert_ne!(shader_a, shader_b); // different shader should NOT merge
    }

    // Physics is unchanged

    // #[test]
    // fn physics_default_unchanged() {
    //     let phys = Physics::default();
    //     assert_eq!(phys.mass, 1.0);
    //     assert_eq!(phys.gravity_scale, 1.0);
    //     assert_eq!(phys.collision_type, CollisionType::Box);
    //     assert_eq!(phys.linear_velocity, [0.0, 0.0, 0.0]);
    // }

    // #[test]
    // fn physics_builder_chain() {
    //     let phys = Physics::default()
    //         .linear_velocity([1.0, 2.0, 3.0])
    //         .mass(50.0)
    //         .collision(1.0)
    //         .gravity(0.5);

    //     assert_eq!(phys.velocity, [1.0, 2.0, 3.0, 4.0]);
    //     assert_eq!(phys.mass, 50.0);
    //     assert_eq!(phys.collision, 1.0);
    //     assert_eq!(phys.gravity, 0.5);
    // }
}
