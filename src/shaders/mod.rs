pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 uv;
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
        layout(location = 4) out vec2 v_uv; 

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
            v_uv = uv;
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
            layout(location = 4) in vec2 v_uv;

            layout(set = 0, binding = 0) uniform UniformBufferObject {
                mat4 view;
                mat4 proj;
                vec3 eye_pos;
                float pad1;
                vec3 light_pos;
                float pad2;
                vec3 light_color;
                float light_intensity;
            } ubo;

            layout(set = 0, binding = 1) uniform sampler2D tex_sampler;

            layout(location = 0) out vec4 f_color;

            void main() {
                vec4 tex_raw = texture(tex_sampler, v_uv);
                vec3 base_color = v_color * tex_raw.rgb;
                
                float roughness = clamp(v_mat_data.x, 0.05, 1.0); // Roughness
                float metalness = v_mat_data.y;                 // Metalness

                vec3 N = normalize(v_normal);
                vec3 V = normalize(ubo.eye_pos - v_pos);
                vec3 L = normalize(ubo.light_pos - v_pos);
                vec3 H = normalize(V + L);

                float dist = length(ubo.light_pos - v_pos);
                float attenuation = ubo.light_intensity / (dist * dist + 1.0);
                vec3 radiance = ubo.light_color * attenuation;

                vec3 f0 = mix(vec3(0.04), base_color, metalness);
                float cosTheta = max(dot(H, V), 0.0);
                vec3 F = f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);

                float ndotl = max(dot(N, L), 0.0);
                float ndotv = max(dot(N, V), 0.0);
                
                float alpha = roughness * roughness;
                float alpha2 = alpha * alpha;
                float ndoth = max(dot(N, H), 0.0);
                float denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0);
                float D = alpha2 / (3.14159 * denom * denom);

                vec3 nominator = D * F;
                float denominator = 4.0 * ndotv * ndotl + 0.001;
                vec3 specular = nominator / denominator;

                vec3 kS = F;
                vec3 kD = (vec3(1.0) - kS) * (1.0 - metalness);
                vec3 diffuse = kD * base_color / 3.14159;

                vec3 result = (diffuse + specular) * radiance * ndotl;
                
                result += vec3(0.03) * base_color;

                result = result / (result + vec3(1.0));
                result = pow(result, vec3(1.0/2.2));

                f_color = vec4(result, tex_raw.a);
            }
        "
    }
}