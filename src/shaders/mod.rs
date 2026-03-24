pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 256) in;

        struct InstanceData {
            mat4 model;
            vec4 color;
            vec4 mat_props;
            vec4 velocity;
            vec4 physic;
        };

        layout(std430, set = 0, binding = 0) readonly buffer ReadBuffer {
            InstanceData data[];
        } read_buf;

        layout(std430, set = 0, binding = 1) writeonly buffer WriteBuffer {
            InstanceData data[];
        } write_buf;

        layout(push_constant) uniform PushConstants {
            float dt;
            uint object_count;
        } pc;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= pc.object_count) return;
            write_buf.data[i].model[0] += 0.0; // its not useless, just trust me, without this shit, nothing will work, bc its not creating a binding, if you delete this, it will crash and say, that there is no binding 1
            InstanceData me = read_buf.data[i];
            vec3 pos = me.model[3].xyz;
            vec3 vel = me.velocity.xyz;
            float radius = me.velocity.w;
            
            float gravity_scale = me.physic.z; 
            float type = me.physic.x;
            float mass = max(me.physic.y, 0.1);
            float dt = pc.dt;

            vel.y -= 90.8 * dt * gravity_scale; 

            float damping = (gravity_scale == 0.0) ? 0.5 : 0.1;
            vel *= (1.0 - damping * dt);
            
            pos += vel * dt;

            for (uint j = 0; j < pc.object_count; ++j) {
                if (i == j) continue;
                InstanceData other = read_buf.data[j];
                
                vec3 o_pos = other.model[3].xyz;
                float o_radius = other.velocity.w;
                float o_type = other.physic.x;
                float o_mass = max(other.physic.y, 0.1);

                vec3 normal = vec3(0.0);
                float overlap = 0.0;

                if (type == 0.0 && o_type == 0.0) { 
                    vec3 delta = pos - o_pos;
                    vec3 dists = abs(delta);
                    vec3 sizes = vec3(radius + o_radius);
                    if (all(lessThan(dists, sizes))) {
                        vec3 face_dist = sizes - dists;
                        if (face_dist.x < face_dist.y && face_dist.x < face_dist.z) {
                            normal = vec3(delta.x > 0.0 ? 1.0 : -1.0, 0.0, 0.0);
                            overlap = face_dist.x;
                        } else if (face_dist.y < face_dist.z) {
                            normal = vec3(0.0, delta.y > 0.0 ? 1.0 : -1.0, 0.0);
                            overlap = face_dist.y;
                        } else {
                            normal = vec3(0.0, 0.0, delta.z > 0.0 ? 1.0 : -1.0);
                            overlap = face_dist.z;
                        }
                    }
                } 
                else if (type == 1.0 && o_type == 1.0) { 
                    vec3 delta = pos - o_pos;
                    float d = length(delta);
                    if (d < radius + o_radius && d > 0.0001) {
                        normal = delta / d;
                        overlap = (radius + o_radius) - d;
                    }
                } 
                else {
                    bool i_am_sphere = (type == 1.0);
                    vec3 s_c = i_am_sphere ? pos : o_pos;
                    vec3 b_c = i_am_sphere ? o_pos : pos;
                    float s_r = i_am_sphere ? radius : o_radius;
                    float b_r = i_am_sphere ? o_radius : radius;
                    vec3 closest = clamp(s_c, b_c - b_r, b_c + b_r);
                    vec3 delta = s_c - closest;
                    float d = length(delta);
                    if (d < s_r) {
                        if (d > 0.0001) {
                            normal = (i_am_sphere ? 1.0 : -1.0) * (delta / d);
                            overlap = s_r - d;
                        } else {
                            vec3 dir = s_c - b_c;
                            float ld = length(dir);
                            normal = (i_am_sphere ? 1.0 : -1.0) * (ld > 0.001 ? dir/ld : vec3(0,1,0));
                            overlap = s_r;
                        }
                    }
                }

                if (overlap > 0.001) {
                    float total_m = mass + o_mass;
                    float m_ratio = o_mass / total_m;

                    pos += normal * overlap * m_ratio * 0.95;

                    vec3 rel_v = vel - other.velocity.xyz;
                    float v_dot = dot(rel_v, normal);
                    if (v_dot < 0.0) {
                        float restitution = (type == 0.0) ? 0.1 : 0.5;
                        float j_mag = -(1.0 + restitution) * v_dot;
                        j_mag /= (1.0/mass + 1.0/o_mass);
                        vel += (j_mag / mass) * normal;

                        vec3 tangent_v = rel_v - (v_dot * normal);
                        vel -= tangent_v * 0.2;
                    }
                }
            }

            // this is floor, uncomment if needed
            // if (pos.y < radius) {
            //     pos.y = radius;
            //     if (vel.y < 0.0) {
            //         vel.y = -vel.y * 0.2;
            //         vel.xz *= 0.8;
            //     }
            // }

            // Write back
            me.model[3] = vec4(pos, 1.0);
            me.velocity.xyz = vel;
            write_buf.data[i] = me;
        }
        ",
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        struct InstanceData {
            mat4 model;
            vec4 color;
            vec4 mat_props;
            vec4 velocity;
            vec4 physic;
        };

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 uv;

        layout(location = 0) out vec3 v_color;
        layout(location = 1) out vec3 v_normal;
        layout(location = 2) out vec3 v_pos;
        layout(location = 3) out vec4 v_mat_data;
        layout(location = 4) out vec2 v_uv;
        layout(location = 5) out float v_emissive;
        layout(location = 6) out vec2 v_screen_uv;

        layout(set = 0, binding = 0) uniform UniformBufferObject {
            mat4 view;
            mat4 proj;
            vec3 eye_pos;
        } ubo;

        layout(std430, set = 0, binding = 2) readonly buffer InstanceBuffer {
            InstanceData instances[];
        };

        void main() {
            InstanceData inst = instances[gl_InstanceIndex];

            vec4 world_pos = inst.model * vec4(position, 1.0);

            gl_Position = ubo.proj * ubo.view * world_pos;

            v_pos = world_pos.xyz;
            v_color = inst.color.xyz;
            v_emissive = inst.color.w;

            mat3 normal_matrix = transpose(inverse(mat3(inst.model)));
            v_normal = normal_matrix * normal;

            v_mat_data = inst.mat_props;
            v_uv = uv;

            v_screen_uv = (gl_Position.xy / gl_Position.w) * 0.5 + 0.5;
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
        layout(location = 5) in float v_emissive;
        layout(location = 6) in vec2 v_screen_uv;

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
            if (v_emissive > 0.5) {
                vec3 result = v_color * 1.5;
                result = vec3(1.0) - exp(-result * 1.2);
                f_color = vec4(result, 1.0);
                return;
            }

            vec4 tex_raw = texture(tex_sampler, v_uv);
            vec3 base_color = v_color * tex_raw.rgb;

            float roughness = clamp(v_mat_data.x, 0.05, 1.0);
            float metalness = v_mat_data.y;

            vec3 N = normalize(v_normal);
            vec3 V = normalize(ubo.eye_pos - v_pos);
            vec3 L = normalize(ubo.light_pos - v_pos);
            vec3 H = normalize(V + L);

            float dist = length(ubo.light_pos - v_pos);
            float attenuation = ubo.light_intensity / (1.0 + 0.005 * dist * dist);
            vec3 radiance = ubo.light_color * attenuation;

            vec3 f0 = mix(vec3(0.04), base_color, metalness);
            float cosTheta = max(dot(H, V), 0.0);
            vec3 F = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);

            float ndotl = max(dot(N, L), 0.0);
            float ndotv = max(dot(N, V), 0.0);

            float alpha = roughness * roughness;
            float alpha2 = alpha * alpha;
            float ndoth = max(dot(N, H), 0.0);
            float denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0);
            float D = alpha2 / (3.14159 * denom * denom);

            vec3 specular = (D * F) / (4.0 * ndotv * ndotl + 0.001);

            vec3 kS = F;
            vec3 kD = (1.0 - kS) * (1.0 - metalness);
            vec3 diffuse = kD * base_color / 3.14159;

            vec3 ambient = 0.05 * base_color;
            vec3 lighting = (diffuse + specular) * radiance * ndotl;

            vec3 result = ambient + lighting;

            float fog_dist = length(ubo.eye_pos - v_pos);
            float fog = exp(-0.002 * fog_dist);
            result = mix(vec3(0.02, 0.02, 0.03), result, fog);

            vec2 center_dist = v_screen_uv - 0.5;
            float vignette = 1.0 - dot(center_dist, center_dist) * 1.2;
            result *= clamp(vignette, 0.0, 1.0);

            result = vec3(1.0) - exp(-result * 1.2);

            f_color = vec4(result, tex_raw.a);
        }
        "
    }
}
