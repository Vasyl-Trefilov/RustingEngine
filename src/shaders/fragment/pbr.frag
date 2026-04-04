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