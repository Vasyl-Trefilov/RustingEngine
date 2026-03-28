pub mod cs1 {
    // this is test shader for fast physic disable
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
            // write_buf.data[i] = me;
        }
        ",
    }
}

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
            vec4 rotation;  
        };

        layout(std430, set = 0, binding = 0) readonly buffer ReadBuffer { InstanceData data[]; } read_buf;
        layout(std430, set = 0, binding = 1) writeonly buffer WriteBuffer { InstanceData data[]; } write_buf;

        layout(push_constant) uniform PushConstants {
            float dt;
            uint object_count;
            uint solid_count;
        } pc;

        mat3 skew(vec3 v) {
            return mat3(
                0.0,  v.z, -v.y,
            -v.z,  0.0,  v.x,
                v.y, -v.x,  0.0
            );
        }

        void main() {
            uint i = gl_GlobalInvocationID.x; // this is id of GPU thread
            if (i >= pc.object_count) return;
            write_buf.data[i].model[0] += 0.0;
            InstanceData me = read_buf.data[i];
            vec3 pos = me.model[3].xyz;
            vec3 vel = me.velocity.xyz;
            vec3 ang_vel = me.rotation.xyz;
            float radius = me.velocity.w; // collision radius
            float type = me.physic.x; // 0 - box, 1 - sphere, 2 - universal
            float mass = max(me.physic.y, 0.01);
            float gravity_scale = me.physic.z;
            float dt = pc.dt; // physic step

            vec3 scale = vec3(length(me.model[0].xyz), length(me.model[1].xyz), length(me.model[2].xyz));
            mat3 rot = mat3(me.model[0].xyz/scale.x, me.model[1].xyz/scale.y, me.model[2].xyz/scale.z); // I do this to get pure rotation, based on every object property. Like that every object will be [-1,1]

            vel.y -= 9.81 * dt * gravity_scale;
            pos += vel * dt; // linear step, so it will be based on time, not fps
            
            if (length(ang_vel) > 0.001) {
                rot += (skew(ang_vel) * rot) * dt;
                vec3 c0 = normalize(rot[0]);
                vec3 c1 = normalize(rot[1] - dot(c0, rot[1]) * c0);
                vec3 c2 = cross(c0, c1);
                rot = mat3(c0, c1, c2);
            }

            for (uint j = 0; j < pc.object_count; ++j) {
                if (i == j) continue;
                InstanceData other = read_buf.data[j];
                if (length(other.model[0].xyz) > 100.0) continue; 

                vec3 o_pos = other.model[3].xyz;
                float o_radius = other.velocity.w;
                float o_type = other.physic.x;

                vec3 normal = vec3(0.0);
                float overlap = 0.0;
                vec3 world_contact = vec3(0.0);

                if (type > 0.5 && o_type > 0.5) { 
                    vec3 delta = pos - o_pos;
                    float d = length(delta);
                    if (d < radius + o_radius) {
                        normal = delta / max(d, 0.001);
                        overlap = (radius + o_radius) - d;
                        world_contact = o_pos + normal * o_radius;
                    }
                } 
                else {
                    bool me_is_box = (type < 0.5);
                    mat3 box_rot = me_is_box ? rot : mat3(other.model[0].xyz/length(other.model[0].xyz), other.model[1].xyz/length(other.model[1].xyz), other.model[2].xyz/length(other.model[2].xyz));
                    vec3 box_pos = me_is_box ? pos : o_pos;
                    vec3 sph_pos = me_is_box ? o_pos : pos;
                    float b_rad = me_is_box ? radius : o_radius;
                    float s_rad = me_is_box ? o_radius : radius;

                    vec3 local_sph = transpose(box_rot) * (sph_pos - box_pos);
                    vec3 local_closest = clamp(local_sph, -vec3(b_rad), vec3(b_rad));
                    vec3 local_delta = local_sph - local_closest;
                    float d = length(local_delta);

                    if (d < s_rad) {
                        overlap = s_rad - d;
                        vec3 local_n = (d > 0.001) ? local_delta / d : vec3(0,1,0);
                        normal = box_rot * local_n; 
                        if (me_is_box) normal = -normal; 
                        world_contact = box_rot * local_closest + box_pos;
                    }
                }

                if (overlap > 0.0) {
                    float o_mass = max(other.physic.y, 0.1);
                    pos += normal * overlap * (o_mass / (mass + o_mass)) * 0.5;

                    vec3 r_me = world_contact - pos;
                    vec3 r_ot = world_contact - o_pos;

                    vec3 v_rel = (vel + cross(ang_vel, r_me)) - (other.velocity.xyz + cross(other.rotation.xyz, r_ot));
                    float v_sep = dot(v_rel, normal);

                    if (v_sep < 0.0) {
                        float i_coeff = (type < 0.5) ? 0.66 : 0.4;
                        float o_coeff = (o_type < 0.5) ? 0.66 : 0.4;
                        float inertia = i_coeff * mass * radius * radius;
                        float o_inertia = o_coeff * o_mass * o_radius * o_radius;
                        
                        float K = (1.0/mass + 1.0/o_mass) + 
                                dot(normal, cross(cross(r_me, normal) / inertia, r_me)) +
                                dot(normal, cross(cross(r_ot, normal) / o_inertia, r_ot));

                        float j_mag = -(1.2 * v_sep) / K; 
                        vec3 impulse = j_mag * normal;

                        vel += impulse / mass;
                        ang_vel += cross(r_me, impulse) / inertia;

                        v_rel = (vel + cross(ang_vel, r_me)) - (other.velocity.xyz + cross(other.rotation.xyz, r_ot));
                        vec3 tangent = v_rel - dot(v_rel, normal) * normal;
                        if (length(tangent) > 0.01) {
                            vec3 t_dir = normalize(tangent);
                            float Kt = (1.0/mass + 1.0/o_mass) + 
                                    dot(t_dir, cross(cross(r_me, t_dir) / inertia, r_me)) +
                                    dot(t_dir, cross(cross(r_ot, t_dir) / o_inertia, r_ot));
                            
                            float friction_mu = 0.8; 
                            float jt_mag = -dot(v_rel, t_dir) / Kt;
                            jt_mag = clamp(jt_mag, -j_mag * friction_mu, j_mag * friction_mu);

                            vec3 f_imp = jt_mag * t_dir;
                            vel += f_imp / mass;
                            ang_vel += cross(r_me, f_imp) / inertia;
                        }
                    }
                }
            }

            if (pos.y < radius) {
                pos.y = radius;
                if (vel.y < 0.0) {
                    vel.y *= -0.2;
                    ang_vel *= 0.8;
                    vel.xz *= 0.9;
                }
            }

            write_buf.data[i].model[0] = vec4(rot[0] * scale.x, 0.0);
            write_buf.data[i].model[1] = vec4(rot[1] * scale.y, 0.0);
            write_buf.data[i].model[2] = vec4(rot[2] * scale.z, 0.0);
            write_buf.data[i].model[3] = vec4(pos, 1.0);
            write_buf.data[i].velocity = vec4(vel, radius);
            write_buf.data[i].rotation = vec4(ang_vel, 0.0);
            write_buf.data[i].color = me.color;
            write_buf.data[i].mat_props = me.mat_props;
            write_buf.data[i].physic = me.physic;
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
            vec4 rotation; 
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
