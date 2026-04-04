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