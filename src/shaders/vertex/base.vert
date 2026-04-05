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
	uint v_base_instance;
} ubo;

layout(push_constant) uniform MeshPush {
    uint v_visible_list_offset;  // 0 for direct rendering, offset for indirect
    uint v_use_culling;           // 0 = direct rendering, 1 = indirect with culling
} mesh_pc;

layout(std430, set = 0, binding = 2) readonly buffer InstanceBuffer { InstanceData instances[]; };
layout(std430, set = 0, binding = 3) readonly buffer VisibleIndices { uint data[]; };

void main() {
    uint actual_id;
    if (mesh_pc.v_use_culling == 1) {
        actual_id = data[gl_InstanceIndex + mesh_pc.v_visible_list_offset];
    } else {
        actual_id = gl_InstanceIndex + mesh_pc.v_visible_list_offset;
    }
    InstanceData inst = instances[actual_id];


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