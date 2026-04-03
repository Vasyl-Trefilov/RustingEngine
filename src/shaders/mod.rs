pub mod cs_basic {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/compute/basic.glsl",
    }
}

pub mod cs_grid_build {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/compute/grid_build.comp",
    }
}

pub mod cs_empty {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/compute/empty.glsl",
    }
}

/// This is maximal available compute shaders with all possible physic functions
pub mod cs_max {
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

        layout(std430, set = 0, binding = 0) readonly buffer ReadBuffer {
            InstanceData data[];
        } read_buf;

        layout(std430, set = 0, binding = 1) writeonly buffer WriteBuffer {
            InstanceData data[];
        } write_buf;

        layout(push_constant) uniform PushConstants {
            float dt;
            uint total_objects;
            uint offset;
            uint count;
        } pc;

        mat3 skew(vec3 v) {
            return mat3(
                0.0,  v.z, -v.y,
               -v.z,  0.0,  v.x,
                v.y, -v.x,  0.0
            );
        }

        void main() {
            uint i = gl_GlobalInvocationID.x + pc.offset;
            if (i >= pc.offset + pc.count) return;

            InstanceData me = read_buf.data[i];
            float mass = me.physic.y;
            if (mass <= 0.0) {
                write_buf.data[i] = me;
                return;
            }

            vec3 pos = me.model[3].xyz;
            vec3 vel = me.velocity.xyz;
            vec3 ang_vel = me.rotation.xyz;
            float type = me.physic.x; 
            float dt = pc.dt;

            vec3 scaleA = vec3(length(me.model[0].xyz), length(me.model[1].xyz), length(me.model[2].xyz));
            vec3 halfA = scaleA * 0.5; 
            mat3 rotA = mat3(me.model[0].xyz / scaleA.x, me.model[1].xyz / scaleA.y, me.model[2].xyz / scaleA.z);

            // Gravity
            vel.y -= 9.81 * me.physic.z * dt;
            pos += vel * dt;

            // Rotation apply
            if (length(ang_vel) > 0.001) {
                rotA += skew(ang_vel) * rotA * dt;
                vec3 c0 = normalize(rotA[0]);
                vec3 c1 = normalize(rotA[1] - dot(c0, rotA[1]) * c0);
                vec3 c2 = cross(c0, c1);
                rotA = mat3(c0, c1, c2);
            }

            // Collision Loop
            for (uint j = 0; j < pc.total_objects; ++j) {
                if (i == j) continue;

                InstanceData other = read_buf.data[j];
                vec3 o_pos = other.model[3].xyz;
                vec3 delta = o_pos - pos;
                
                vec3 o_scale = vec3(length(other.model[0].xyz), length(other.model[1].xyz), length(other.model[2].xyz));
                vec3 halfB = o_scale * 0.5;
                float o_type = other.physic.x;

                // This is used to correct sphere collision
                float boundA = (type > 0.5) ? scaleA.x : length(halfA);
                float boundB = (o_type > 0.5) ? o_scale.x : length(halfB);
                if (dot(delta, delta) > pow(boundA + boundB, 2.0)) continue;

                mat3 rotB = mat3(other.model[0].xyz / o_scale.x, other.model[1].xyz / o_scale.y, other.model[2].xyz / o_scale.z);
                vec3 normal = vec3(0.0);
                float overlap = 0.0;
                vec3 world_contact = vec3(0.0);

                // Sphere - Sphere
                if (type > 0.5 && o_type > 0.5) {
                    float d = length(delta);
                    float sum_r = scaleA.x + o_scale.x; 
                    if (d < sum_r) {
                        normal = -delta / max(d, 0.0001); 
                        overlap = sum_r - d;
                        world_contact = pos - normal * scaleA.x;
                    }
                }
                // Box - Box
                else if (type < 0.5 && o_type < 0.5) {
                    float min_overlap = 1e9;
                    vec3 best_axis;
                    bool separating = false;
                    vec3 axes[15];
                    axes[0] = rotA[0]; axes[1] = rotA[1]; axes[2] = rotA[2];
                    axes[3] = rotB[0]; axes[4] = rotB[1]; axes[5] = rotB[2];
                    axes[6] = cross(rotA[0], rotB[0]); axes[7] = cross(rotA[0], rotB[1]); axes[8] = cross(rotA[0], rotB[2]);
                    axes[9] = cross(rotA[1], rotB[0]); axes[10] = cross(rotA[1], rotB[1]); axes[11] = cross(rotA[1], rotB[2]);
                    axes[12] = cross(rotA[2], rotB[0]); axes[13] = cross(rotA[2], rotB[1]); axes[14] = cross(rotA[2], rotB[2]);

                    for (int a = 0; a < 15; a++) {
                        vec3 L = axes[a];
                        float lenSq = dot(L, L);
                        if (lenSq < 1e-6) continue; 
                        L *= inversesqrt(lenSq); 
                        float rA = halfA.x * abs(dot(rotA[0], L)) + halfA.y * abs(dot(rotA[1], L)) + halfA.z * abs(dot(rotA[2], L));
                        float rB = halfB.x * abs(dot(rotB[0], L)) + halfB.y * abs(dot(rotB[1], L)) + halfB.z * abs(dot(rotB[2], L));
                        float s = rA + rB - abs(dot(delta, L));
                        if (s <= 0.0) { separating = true; break; }
                        if (s < min_overlap) { min_overlap = s; best_axis = L; }
                    }
                    if (!separating) {
                        overlap = min_overlap;
                        normal = (dot(delta, best_axis) > 0.0) ? -best_axis : best_axis;
                        
                        // Check if the normal is aligned with any of our local faces
                        vec3 local_n = transpose(rotA) * (-normal);
                        if (abs(local_n.x) > 0.98) world_contact = pos + rotA[0] * (local_n.x * halfA.x);
                        else if (abs(local_n.y) > 0.98) world_contact = pos + rotA[1] * (local_n.y * halfA.y);
                        else if (abs(local_n.z) > 0.98) world_contact = pos + rotA[2] * (local_n.z * halfA.z);
                        else {
                            // Hit a corner or edge
                            vec3 c_local = vec3(
                                (local_n.x > 0.0) ? halfA.x : -halfA.x,
                                (local_n.y > 0.0) ? halfA.y : -halfA.y,
                                (local_n.z > 0.0) ? halfA.z : -halfA.z
                            );
                            world_contact = pos + rotA * c_local;
                        }
                    }
                }
                // Box - Sphere
                else { 
                    bool i_is_box = (type < 0.5);
                    vec3 b_pos = i_is_box ? pos : o_pos;
                    vec3 s_pos = i_is_box ? o_pos : pos;
                    mat3 b_rot = i_is_box ? rotA : rotB;
                    vec3 b_half = i_is_box ? halfA : halfB;
                    float s_rad = i_is_box ? o_scale.x : scaleA.x;
                    vec3 local_s = transpose(b_rot) * (s_pos - b_pos);
                    vec3 closest = clamp(local_s, -b_half, b_half);
                    vec3 local_delta = local_s - closest;
                    float d = length(local_delta);

                    if (d < s_rad && d > 0.0001) {
                        overlap = s_rad - d;
                        normal = b_rot * (local_delta / d);
                        if (i_is_box) normal = -normal;
                        world_contact = b_rot * closest + b_pos;
                    }
                }

                if (overlap > 0.0) {
                    float o_mass = max(other.physic.y, 0.001);
                    float my_m = mass;
                    float ot_m = o_mass;

                    // If I am on top of the other object, make the other object essentially unmovable so I get pushed up.
                    if (pos.y > o_pos.y + 0.1) { ot_m *= 10.0; }
                    else if (o_pos.y > pos.y + 0.1) { my_m *= 10.0; }

                    float total_m = my_m + ot_m;
                    float ratio = ot_m / total_m;
                    pos += normal * overlap * ratio * 0.95; // High correction factor

                    // Impulse Math
                    vec3 r_me = world_contact - pos;
                    vec3 r_ot = world_contact - o_pos;
                    float inertia = (type < 0.5) ? mass * dot(scaleA, scaleA) / 6.0 : 0.4 * mass * scaleA.x * scaleA.x;
                    float o_inertia = (o_type < 0.5) ? o_mass * dot(o_scale, o_scale) / 6.0 : 0.4 * o_mass * o_scale.x * o_scale.x;

                    vec3 v_rel = (vel + cross(ang_vel, r_me)) - (other.velocity.xyz + cross(other.rotation.xyz, r_ot));
                    float v_sep = dot(v_rel, normal);

                    if (v_sep < 0.0) {
                        float K = (1.0/mass + 1.0/o_mass) + dot(normal, cross(cross(r_me, normal)/inertia, r_me)) + dot(normal, cross(cross(r_ot, normal)/o_inertia, r_ot));
                        float j = -(1.1 * v_sep) / K; 
                        vec3 impulse = j * normal;

                        vel += impulse / mass;
                        
                        // Only add rotation if we aren't hitting the face center or if movement is fast
                        if (overlap > 0.01 || abs(v_sep) > 0.1) {
                             ang_vel += cross(r_me, impulse) / inertia;
                        }

                        // Friction
                        v_rel = (vel + cross(ang_vel, r_me)) - (other.velocity.xyz + cross(other.rotation.xyz, r_ot));
                        vec3 tangent = v_rel - dot(v_rel, normal) * normal;
                        if (length(tangent) > 0.01) {
                            vec3 t_dir = normalize(tangent);
                            float Kt = (1.0/mass + 1.0/o_mass) + dot(t_dir, cross(cross(r_me, t_dir)/inertia, r_me)) + dot(t_dir, cross(cross(r_ot, t_dir)/o_inertia, r_ot));
                            float jt = clamp(-dot(v_rel, t_dir) / Kt, -j * 0.5, j * 0.5);
                            vec3 f_imp = jt * t_dir;
                            vel += f_imp / mass;
                            ang_vel += cross(r_me, f_imp) / inertia;
                        }
                    }
                }
            }

            // Shader floor. Uses as backup
            float lowest_y = (type > 0.5) ? scaleA.x : (halfA.x * abs(rotA[0].y) + halfA.y * abs(rotA[1].y) + halfA.z * abs(rotA[2].y));
            if (pos.y < lowest_y) {
                pos.y = lowest_y;
                if (vel.y < 0.0) {
                    vel.y *= -0.05; // Low bounce for stability
                    vel.xz *= 0.8;
                    ang_vel *= 0.7;
                    
                    // Tip over logic
                    if (type < 0.5 && rotA[1].y < 0.99) {
                        vec3 local_down = transpose(rotA) * vec3(0, -1, 0);
                        vec3 edge = rotA * vec3((local_down.x > 0.0) ? halfA.x : -halfA.x, (local_down.y > 0.0) ? halfA.y : -halfA.y, (local_down.z > 0.0) ? halfA.z : -halfA.z);
                        ang_vel += cross(edge, vec3(0, 1, 0)) * 2.0 * dt;
                    }
                }
            }
            
            // Sleep logic to avoid infinity moving
            if (length(vel) < 0.02 && length(ang_vel) < 0.02) {
                vel = vec3(0,0,0);
                ang_vel = vec3(0,0,0);
            }

            // Write back
            write_buf.data[i].model[0] = vec4(rotA[0] * scaleA.x, 0.0);
            write_buf.data[i].model[1] = vec4(rotA[1] * scaleA.y, 0.0);
            write_buf.data[i].model[2] = vec4(rotA[2] * scaleA.z, 0.0);
            write_buf.data[i].model[3] = vec4(pos, 1.0);
            write_buf.data[i].velocity = vec4(vel, me.velocity.w);
            write_buf.data[i].rotation = vec4(ang_vel, 0.0);
            write_buf.data[i].color = me.color;
            write_buf.data[i].mat_props = me.mat_props;
            write_buf.data[i].physic = me.physic;
        }
        "
    }
}

/// This is maximal available compute shaders with all possible physic functions
pub mod cs_full {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
       #version 450

layout(local_size_x = 256) in;

// NEW INSTANCE DATA
struct InstanceData {
    mat4 model;
    vec4 color;
    vec4 mat_props;
    vec4 velocity;          // xyz = vel, w = bounciness
    vec4 angular_velocity;  // xyz = ang_vel, w = friction
    vec4 physic_props;      // x = type, y = mass, z = gravity_scale, w = grid_hack
};

layout(std430, set = 0, binding = 0) readonly buffer ReadBuffer {
    InstanceData data[];
} read_buf;

layout(std430, set = 0, binding = 1) writeonly buffer WriteBuffer {
    InstanceData data[];
} write_buf;

layout(push_constant) uniform PushConstants {
    float dt;
    uint total_objects;
    uint offset;
    uint count;
    uint num_big_objects;
    uint _pad[3];
    vec4 global_gravity;
} pc;

mat3 skew(vec3 v) {
    return mat3(
        0.0,  v.z, -v.y,
       -v.z,  0.0,  v.x,
        v.y, -v.x,  0.0
    );
}

void main() {
    uint i = gl_GlobalInvocationID.x + pc.offset;
    if (i >= pc.offset + pc.count) return;

    InstanceData me = read_buf.data[i];

    float mass = me.physic_props.y;
    if (mass <= 0.0) {
        write_buf.data[i] = me;
        return;
    }

    vec3 pos = me.model[3].xyz;
    vec3 vel = me.velocity.xyz;
    
    vec3 ang_vel = me.angular_velocity.xyz;
    
    float type = me.physic_props.x; 
    float dt = pc.dt;

    vec3 scaleA = vec3(length(me.model[0].xyz), length(me.model[1].xyz), length(me.model[2].xyz));
    vec3 halfA = scaleA * 0.5; 
    mat3 rotA = mat3(me.model[0].xyz / scaleA.x, me.model[1].xyz / scaleA.y, me.model[2].xyz / scaleA.z);

    vel.y -= 9.81 * me.physic_props.z * dt;
    pos += vel * dt;

    if (length(ang_vel) > 0.001) {
        rotA += skew(ang_vel) * rotA * dt;
        vec3 c0 = normalize(rotA[0]);
        vec3 c1 = normalize(rotA[1] - dot(c0, rotA[1]) * c0);
        vec3 c2 = cross(c0, c1);
        rotA = mat3(c0, c1, c2);
    }

    for (uint j = 0; j < pc.total_objects; ++j) {
        if (i == j) continue;

        InstanceData other = read_buf.data[j];
        vec3 o_pos = other.model[3].xyz;
        vec3 delta = o_pos - pos;
        
        vec3 o_scale = vec3(length(other.model[0].xyz), length(other.model[1].xyz), length(other.model[2].xyz));
        vec3 halfB = o_scale * 0.5;
        
        float o_type = other.physic_props.x;

        float boundA = (type > 0.5) ? scaleA.x : length(halfA);
        float boundB = (o_type > 0.5) ? o_scale.x : length(halfB);
        if (dot(delta, delta) > pow(boundA + boundB, 2.0)) continue;

        mat3 rotB = mat3(other.model[0].xyz / o_scale.x, other.model[1].xyz / o_scale.y, other.model[2].xyz / o_scale.z);
        vec3 normal = vec3(0.0);
        float overlap = 0.0;
        vec3 world_contact = vec3(0.0);

        // Sphere - Sphere
        if (type > 0.5 && o_type > 0.5) {
            float d = length(delta);
            float sum_r = scaleA.x + o_scale.x; 
            if (d < sum_r) {
                normal = -delta / max(d, 0.0001); 
                overlap = sum_r - d;
                world_contact = pos - normal * scaleA.x;
            }
        }
        // Box - Box
        else if (type < 0.5 && o_type < 0.5) {
            float min_overlap = 1e9;
            vec3 best_axis;
            bool separating = false;
            vec3 axes[15];
            axes[0] = rotA[0]; axes[1] = rotA[1]; axes[2] = rotA[2];
            axes[3] = rotB[0]; axes[4] = rotB[1]; axes[5] = rotB[2];
            axes[6] = cross(rotA[0], rotB[0]); axes[7] = cross(rotA[0], rotB[1]); axes[8] = cross(rotA[0], rotB[2]);
            axes[9] = cross(rotA[1], rotB[0]); axes[10] = cross(rotA[1], rotB[1]); axes[11] = cross(rotA[1], rotB[2]);
            axes[12] = cross(rotA[2], rotB[0]); axes[13] = cross(rotA[2], rotB[1]); axes[14] = cross(rotA[2], rotB[2]);

            for (int a = 0; a < 15; a++) {
                vec3 L = axes[a];
                float lenSq = dot(L, L);
                if (lenSq < 1e-6) continue; 
                L *= inversesqrt(lenSq); 
                float rA = halfA.x * abs(dot(rotA[0], L)) + halfA.y * abs(dot(rotA[1], L)) + halfA.z * abs(dot(rotA[2], L));
                float rB = halfB.x * abs(dot(rotB[0], L)) + halfB.y * abs(dot(rotB[1], L)) + halfB.z * abs(dot(rotB[2], L));
                float s = rA + rB - abs(dot(delta, L));
                if (s <= 0.0) { separating = true; break; }
                if (s < min_overlap) { min_overlap = s; best_axis = L; }
            }
            if (!separating) {
                overlap = min_overlap;
                normal = (dot(delta, best_axis) > 0.0) ? -best_axis : best_axis;
                
                vec3 local_n = transpose(rotA) * (-normal);
                if (abs(local_n.x) > 0.98) world_contact = pos + rotA[0] * (local_n.x * halfA.x);
                else if (abs(local_n.y) > 0.98) world_contact = pos + rotA[1] * (local_n.y * halfA.y);
                else if (abs(local_n.z) > 0.98) world_contact = pos + rotA[2] * (local_n.z * halfA.z);
                else {
                    vec3 c_local = vec3(
                        (local_n.x > 0.0) ? halfA.x : -halfA.x,
                        (local_n.y > 0.0) ? halfA.y : -halfA.y,
                        (local_n.z > 0.0) ? halfA.z : -halfA.z
                    );
                    world_contact = pos + rotA * c_local;
                }
            }
        }
        // Box - Sphere
        else { 
            bool i_is_box = (type < 0.5);
            vec3 b_pos = i_is_box ? pos : o_pos;
            vec3 s_pos = i_is_box ? o_pos : pos;
            mat3 b_rot = i_is_box ? rotA : rotB;
            vec3 b_half = i_is_box ? halfA : halfB;
            float s_rad = i_is_box ? o_scale.x : scaleA.x;
            vec3 local_s = transpose(b_rot) * (s_pos - b_pos);
            vec3 closest = clamp(local_s, -b_half, b_half);
            vec3 local_delta = local_s - closest;
            float d = length(local_delta);

            if (d < s_rad && d > 0.0001) {
                overlap = s_rad - d;
                normal = b_rot * (local_delta / d);
                if (i_is_box) normal = -normal;
                world_contact = b_rot * closest + b_pos;
            }
        }

        if (overlap > 0.0) {
            float o_mass = max(other.physic_props.y, 0.001);
            float my_m = mass;
            float ot_m = o_mass;

            if (pos.y > o_pos.y + 0.1) { ot_m *= 10.0; }
            else if (o_pos.y > pos.y + 0.1) { my_m *= 10.0; }

            float total_m = my_m + ot_m;
            float ratio = ot_m / total_m;
            pos += normal * overlap * ratio * 0.95; 

            vec3 r_me = world_contact - pos;
            vec3 r_ot = world_contact - o_pos;
            float inertia = (type < 0.5) ? mass * dot(scaleA, scaleA) / 6.0 : 0.4 * mass * scaleA.x * scaleA.x;
            float o_inertia = (o_type < 0.5) ? o_mass * dot(o_scale, o_scale) / 6.0 : 0.4 * o_mass * o_scale.x * o_scale.x;

            vec3 v_rel = (vel + cross(ang_vel, r_me)) - (other.velocity.xyz + cross(other.angular_velocity.xyz, r_ot));
            float v_sep = dot(v_rel, normal);

            if (v_sep < 0.0) {
                float K = (1.0/mass + 1.0/o_mass) + dot(normal, cross(cross(r_me, normal)/inertia, r_me)) + dot(normal, cross(cross(r_ot, normal)/o_inertia, r_ot));
                float j = -(1.1 * v_sep) / K; 
                vec3 impulse = j * normal;

                vel += impulse / mass;
                
                if (overlap > 0.01 || abs(v_sep) > 0.1) {
                     ang_vel += cross(r_me, impulse) / inertia;
                }

                v_rel = (vel + cross(ang_vel, r_me)) - (other.velocity.xyz + cross(other.angular_velocity.xyz, r_ot));
                vec3 tangent = v_rel - dot(v_rel, normal) * normal;
                if (length(tangent) > 0.01) {
                    vec3 t_dir = normalize(tangent);
                    float Kt = (1.0/mass + 1.0/o_mass) + dot(t_dir, cross(cross(r_me, t_dir)/inertia, r_me)) + dot(t_dir, cross(cross(r_ot, t_dir)/o_inertia, r_ot));
                    float jt = clamp(-dot(v_rel, t_dir) / Kt, -j * 0.5, j * 0.5);
                    vec3 f_imp = jt * t_dir;
                    vel += f_imp / mass;
                    ang_vel += cross(r_me, f_imp) / inertia;
                }
            }
        }
    }

    float lowest_y = (type > 0.5) ? scaleA.x : (halfA.x * abs(rotA[0].y) + halfA.y * abs(rotA[1].y) + halfA.z * abs(rotA[2].y));
    if (pos.y < lowest_y) {
        pos.y = lowest_y;
        if (vel.y < 0.0) {
            vel.y *= -0.05; 
            vel.xz *= 0.8;
            ang_vel *= 0.7;
            
            if (type < 0.5 && rotA[1].y < 0.99) {
                vec3 local_down = transpose(rotA) * vec3(0, -1, 0);
                vec3 edge = rotA * vec3((local_down.x > 0.0) ? halfA.x : -halfA.x, (local_down.y > 0.0) ? halfA.y : -halfA.y, (local_down.z > 0.0) ? halfA.z : -halfA.z);
                ang_vel += cross(edge, vec3(0, 1, 0)) * 2.0 * dt;
            }
        }
    }
    
    if (length(vel) < 0.02 && length(ang_vel) < 0.02) {
        vel = vec3(0,0,0);
        ang_vel = vec3(0,0,0);
    }

    write_buf.data[i].model[0] = vec4(rotA[0] * scaleA.x, 0.0);
    write_buf.data[i].model[1] = vec4(rotA[1] * scaleA.y, 0.0);
    write_buf.data[i].model[2] = vec4(rotA[2] * scaleA.z, 0.0);
    write_buf.data[i].model[3] = vec4(pos, 1.0);
    
    write_buf.data[i].velocity = vec4(vel, me.velocity.w);
    write_buf.data[i].angular_velocity = vec4(ang_vel, me.angular_velocity.w); 
    
    write_buf.data[i].color = me.color;
    write_buf.data[i].mat_props = me.mat_props;
    write_buf.data[i].physic_props = me.physic_props;
}
        "
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

        layout(std430, set = 0, binding = 0) readonly buffer ReadBuffer {
            InstanceData data[];
        } read_buf;

        layout(std430, set = 0, binding = 1) writeonly buffer WriteBuffer {
            InstanceData data[];
        } write_buf;

        layout(push_constant) uniform PushConstants {
            float dt;
            uint total_objects;
            uint offset;
            uint count;
        } pc;

        mat3 skew(vec3 v) {
            return mat3(
                0.0,  v.z, -v.y,
               -v.z,  0.0,  v.x,
                v.y, -v.x,  0.0
            );
        }

        void main() {
            uint i = gl_GlobalInvocationID.x + pc.offset;
            if (i >= pc.offset + pc.count) return;

            InstanceData me = read_buf.data[i];
            float mass = me.physic.y;
            if (mass <= 0.0) {
                write_buf.data[i] = me;
                return;
            }

            vec3 pos = me.model[3].xyz;
            vec3 vel = me.velocity.xyz;
            vec3 ang_vel = me.rotation.xyz;
            float type = me.physic.x; 
            float dt = pc.dt;

            vec3 scaleA = vec3(length(me.model[0].xyz), length(me.model[1].xyz), length(me.model[2].xyz));
            vec3 halfA = scaleA * 0.5; 
            mat3 rotA = mat3(me.model[0].xyz / scaleA.x, me.model[1].xyz / scaleA.y, me.model[2].xyz / scaleA.z);

            vel.y -= 9.81 * me.physic.z * dt;
            pos += vel * dt;

            if (length(ang_vel) > 0.001) {
                rotA += skew(ang_vel) * rotA * dt;
                vec3 c0 = normalize(rotA[0]);
                vec3 c1 = normalize(rotA[1] - dot(c0, rotA[1]) * c0);
                vec3 c2 = cross(c0, c1);
                rotA = mat3(c0, c1, c2);
            }

            // ang_vel *= 0.98; 
            
            for (uint j = 0; j < pc.total_objects; ++j) {
                if (i == j) continue;

                InstanceData other = read_buf.data[j];
                vec3 o_pos = other.model[3].xyz;
                vec3 delta = o_pos - pos;
                
                vec3 o_scale = vec3(length(other.model[0].xyz), length(other.model[1].xyz), length(other.model[2].xyz));
                vec3 halfB = o_scale * 0.5;
                
                float o_type = other.physic.x;

                float boundA = (type > 0.5) ? scaleA.x : length(halfA);
                float boundB = (o_type > 0.5) ? o_scale.x : length(halfB);
                
                float max_dist = boundA + boundB;
                if (dot(delta, delta) > max_dist * max_dist) continue;


                mat3 rotB = mat3(other.model[0].xyz / o_scale.x, other.model[1].xyz / o_scale.y, other.model[2].xyz / o_scale.z);

                vec3 normal = vec3(0.0);
                float overlap = 0.0;

                // SPHERE-SPHERE
                if (type > 0.5 && o_type > 0.5) {
                    float d = length(delta);
                    // spheres use full scale.x as their radius
                    float sum_r = scaleA.x + o_scale.x; 
                    if (d < sum_r && d > 0.0001) {
                        normal = -delta / d; 
                        overlap = sum_r - d;
                    } else if (d <= 0.0001) {
                        normal = vec3(0, 1, 0);
                        overlap = sum_r;
                    }
                }
                // BOX-BOX
                else if (type < 0.5 && o_type < 0.5) {
                    float min_overlap = 1e9;
                    vec3 best_axis;
                    bool separating = false;

                    vec3 axes[15];
                    axes[0] = rotA[0]; axes[1] = rotA[1]; axes[2] = rotA[2];
                    axes[3] = rotB[0]; axes[4] = rotB[1]; axes[5] = rotB[2];
                    axes[6] = cross(rotA[0], rotB[0]); axes[7] = cross(rotA[0], rotB[1]); axes[8] = cross(rotA[0], rotB[2]);
                    axes[9] = cross(rotA[1], rotB[0]); axes[10] = cross(rotA[1], rotB[1]); axes[11] = cross(rotA[1], rotB[2]);
                    axes[12] = cross(rotA[2], rotB[0]); axes[13] = cross(rotA[2], rotB[1]); axes[14] = cross(rotA[2], rotB[2]);

                    for (int a = 0; a < 15; a++) {
                        vec3 L = axes[a];
                        float lenSq = dot(L, L);
                        if (lenSq < 1e-6) continue; 
                        L *= inversesqrt(lenSq); 

                        float dist = abs(dot(delta, L));
                        float rA = halfA.x * abs(dot(rotA[0], L)) + halfA.y * abs(dot(rotA[1], L)) + halfA.z * abs(dot(rotA[2], L));
                        float rB = halfB.x * abs(dot(rotB[0], L)) + halfB.y * abs(dot(rotB[1], L)) + halfB.z * abs(dot(rotB[2], L));
                        float s = rA + rB - dist;

                        if (s <= 0.0) { separating = true; break; }
                        if (s < min_overlap) { min_overlap = s; best_axis = L; }
                    }

                    if (!separating) {
                        overlap = min_overlap;
                        normal = (dot(delta, best_axis) > 0.0) ? -best_axis : best_axis;
                    }
                }
                // BOX-SPHERE
                else {
                    bool i_is_box = (type < 0.5);
                    vec3 b_pos = i_is_box ? pos : o_pos;
                    vec3 s_pos = i_is_box ? o_pos : pos;
                    mat3 b_rot = i_is_box ? rotA : rotB;
                    vec3 b_half = i_is_box ? halfA : halfB;
                    
                    float s_rad = i_is_box ? o_scale.x : scaleA.x;

                    vec3 local_s = transpose(b_rot) * (s_pos - b_pos);
                    vec3 closest = clamp(local_s, -b_half, b_half);
                    vec3 local_delta = local_s - closest;
                    float d = length(local_delta);

                    if (d < s_rad && d > 0.0001) {
                        overlap = s_rad - d;
                        normal = b_rot * (local_delta / d);
                        if (i_is_box) normal = -normal;
                    } else if (d <= 0.0001) { 
                        overlap = s_rad;
                        normal = vec3(0, 1, 0);
                        if (i_is_box) normal = -normal;
                    }
                }

                // Impulse Math
                if (overlap > 0.0) {
                    float my_mass = mass;
                    float their_mass = other.physic.y;

                    if (my_mass > 0.0 && their_mass > 0.0) {
                        if (pos.y < o_pos.y - 0.2) {
                            my_mass *= 5.0;     
                        } else if (pos.y > o_pos.y + 0.2) {
                            their_mass *= 5.0;  
                        }
                    }

                    float total_mass = (their_mass <= 0.0) ? my_mass : my_mass + their_mass;
                    float ratio = (their_mass <= 0.0) ? 1.0 : their_mass / total_mass;

                    // Increased clamp limit so giant spheres can properly shove cubes out of the way
                    overlap = min(overlap, 3.0); 

                    pos += normal * overlap * ratio * 0.9;

                    vec3 o_vel = other.velocity.xyz;
                    vec3 relative_vel = vel - o_vel;
                    float v_rel = dot(relative_vel, normal);
                    
                    if (v_rel < 0.0) {
                        float bounce = (abs(v_rel) < 1.0) ? 0.0 : 0.2;
                        
                        // Properly transfer momentum based on the mass ratio
                        vel -= normal * v_rel * (1.0 + bounce) * ratio; 
                        
                        vec3 tangent = relative_vel - normal * v_rel;
                        vel -= tangent * 0.5 * ratio;
                        ang_vel *= 0.9; 
                    }
                }
            }

            if (pos.y < halfA.y) {
                pos.y = halfA.y;
                if (vel.y < 0.0) {
                    vel.y *= -0.1;
                    vec3 tangent = vel - vec3(0, vel.y, 0);
                    vel -= tangent * 0.5;
                    ang_vel *= 0.5;
                }
            }

            write_buf.data[i].model[0] = vec4(rotA[0] * scaleA.x, 0.0);
            write_buf.data[i].model[1] = vec4(rotA[1] * scaleA.y, 0.0);
            write_buf.data[i].model[2] = vec4(rotA[2] * scaleA.z, 0.0);
            write_buf.data[i].model[3] = vec4(pos, 1.0);
            write_buf.data[i].velocity = vec4(vel, me.velocity.w);
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

pub mod fs_unlit {
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
            // Dummy usage to keep UBO layout matching the PBR pipeline
            float dummy = (ubo.light_intensity + ubo.light_pos.x + ubo.light_color.x + ubo.view[0][0]) * 0.0000001;

            vec4 tex_raw = texture(tex_sampler, v_uv);
            vec3 base_color = v_color * tex_raw.rgb;
            f_color = vec4(base_color + vec3(dummy), tex_raw.a);
        }
        "
    }
}

pub mod fs_emissive {
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
            // Dummy usage to keep UBO layout matching the PBR pipeline
            float dummy = (ubo.light_intensity + ubo.light_pos.x + ubo.light_color.x + ubo.view[0][0]) * 0.0000001;

            vec4 tex_raw = texture(tex_sampler, v_uv);
            vec3 base_color = v_color * tex_raw.rgb;

            // Emissive glow with tone mapping
            vec3 result = base_color * 1.5;
            result = vec3(1.0) - exp(-result * 1.2);

            // Vignette
            vec2 center_dist = v_screen_uv - 0.5;
            float vignette = 1.0 - dot(center_dist, center_dist) * 1.2;
            result *= clamp(vignette, 0.0, 1.0);

            f_color = vec4(result + vec3(dummy), tex_raw.a);
        }
        "
    }
}

pub mod fs_normal_debug {
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
            // Dummy usage to keep UBO layout matching the PBR pipeline
            float dummy = (ubo.light_intensity + ubo.light_pos.x + ubo.light_color.x + ubo.view[0][0]) * 0.0000001;

            vec3 N = normalize(v_normal);
            // Map normal from [-1,1] to [0,1] for visualization
            vec3 result = N * 0.5 + 0.5;
            f_color = vec4(result + vec3(dummy), 1.0);
        }
        "
    }
}

pub mod cs2 {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450
        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

        struct InstanceData {
            mat4 model;
            vec4 color;
            vec4 mat_props;
            vec4 velocity;
            vec4 physic; 
            vec4 rotation;
        };

        layout(set = 0, binding = 0) buffer ReadBuf {
            InstanceData data[];
        } read_buf;

        layout(set = 0, binding = 1) buffer WriteBuf {
            InstanceData data[];
        } write_buf;

        layout(push_constant) uniform PushConstants {
            float dt;
            uint total_objects;
            uint offset;
            uint count;
        } pc;

        mat3 skew(vec3 v) {
            return mat3(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0);
        }

        void main() {
            uint i = gl_GlobalInvocationID.x + pc.offset;
            if (i >= pc.offset + pc.count) return;
            
            write_buf.data[i] = read_buf.data[i];
            
            InstanceData me = read_buf.data[i];
            vec3 pos = me.model[3].xyz;
            vec3 vel = me.velocity.xyz;
            vec3 ang_vel = me.rotation.xyz;
            
            float radius = me.velocity.w;
            float type = me.physic.x;
            float mass = me.physic.y;
            float bounce = me.physic.z;
            float grav = me.physic.w;

            if (mass <= 0.0) return;

            vel += vec3(0.0, -9.81 * grav * pc.dt, 0.0);

            // Floor collision
            float ground_level = 0.5 + radius;
            if (pos.y < ground_level) {
                pos.y = ground_level;
                vel.y = -vel.y * bounce;
                vel.x *= 0.98;
                vel.z *= 0.98;
                ang_vel *= 0.95;
            }

            pos += vel * pc.dt;

            mat3 rot = mat3(me.model[0].xyz, me.model[1].xyz, me.model[2].xyz);
            if (length(ang_vel) > 0.001) {
                rot += skew(ang_vel) * rot * pc.dt;
                vec3 c0 = normalize(rot[0]);
                vec3 c1 = normalize(rot[1] - dot(c0, rot[1]) * c0);
                vec3 c2 = cross(c0, c1);
                rot = mat3(c0, c1, c2);
            }

            me.model[0].xyz = rot[0];
            me.model[1].xyz = rot[1];
            me.model[2].xyz = rot[2];
            me.model[3].xyz = pos;

            me.velocity.xyz = vel;
            me.rotation.xyz = ang_vel;

            write_buf.data[i] = me;
        }
        ",
    }
}
