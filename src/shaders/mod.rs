pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec3 normal;      
        layout(location = 3) in vec3 barycentric; 
        layout(location = 4) in vec4 model_row0;
        layout(location = 5) in vec4 model_row1;
        layout(location = 6) in vec4 model_row2;
        layout(location = 7) in vec4 model_row3;
        layout(location = 8) in vec3 instance_color; 
        layout(location = 9) in vec4 instance_mat_props;

        layout(location = 0) out vec3 v_color;
        layout(location = 1) out vec3 v_normal;
        layout(location = 2) out vec3 v_pos;
        layout(location = 3) out vec4 v_mat_data; 

        layout(set = 0, binding = 0) uniform UniformBufferObject {
            mat4 view;
            mat4 proj;
            vec3 eye_pos;
        } ubo;

        void main() {
            mat4 instance_model = mat4(model_row0, model_row1, model_row2, model_row3);
            vec4 world_pos = instance_model * vec4(position, 1.0);
            gl_Position = ubo.proj * ubo.view * world_pos;
            v_pos = world_pos.xyz; 
            v_color = instance_color; 
            v_normal = mat3(instance_model) * normal; 
            v_mat_data = instance_mat_props;
        }
        "
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec3 v_color;
            layout(location = 1) in vec3 v_normal;
            layout(location = 2) in vec3 v_pos;
            layout(location = 3) in vec4 v_mat_data;

            layout(set = 0, binding = 0) uniform UniformBufferObject {
                mat4 view;
                mat4 proj;
                vec3 eye_pos;
                vec3 light_pos;
                vec3 light_color;   
                float light_intensity;
            } ubo;

            layout(location = 0) out vec4 f_color;

            void main() {
            float shininess = v_mat_data.x;
            float spec_strength = v_mat_data.y;
            float roughness = v_mat_data.z;
            float metalness = v_mat_data.w;

            vec3 norm = normalize(v_normal);
            vec3 light_vec = ubo.light_pos - v_pos;
            float distance = length(light_vec);
            vec3 light_dir = normalize(light_vec);
            vec3 view_dir = normalize(ubo.eye_pos - v_pos);
            vec3 halfway_dir = normalize(light_dir + view_dir);

            float attenuation = ubo.light_intensity / (distance * distance + 1.0);

            // 1. Diffuse
            float metal_diffuse_factor = mix(1.0, 0.2, metalness); 
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * ubo.light_color * metal_diffuse_factor;

            // 2. Specular
            vec3 spec_color = mix(vec3(1.0), v_color, metalness); 
            float spec_angle = max(dot(norm, halfway_dir), 0.0);
            float spec_factor = pow(spec_angle, shininess) * (1.0 - roughness);
            vec3 specular = spec_strength * spec_factor * ubo.light_color * spec_color;

            // 3. Ambient
            float ambient_boost = mix(0.03, 0.08, metalness);
            vec3 ambient = ambient_boost * v_color;

            // 4. Combine
            vec3 result = ambient + (diffuse * v_color + specular) * attenuation;

            // Tone Mapping
            result = result / (result + vec3(1.0));
            
            f_color = vec4(result, 1.0);
        }
        "
    }
}