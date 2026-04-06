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

float hash(vec3 p) {
	p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3D(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);

	return mix(
		mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
		    mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
		mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
		    mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
		f.z);
}

float fbm(vec3 p) {
	float value = 0.0;
	float amplitude = 0.5;
	float frequency = 1.0;
	for (int i = 0; i < 6; i++) {
		value += amplitude * noise3D(p * frequency);
		amplitude *= 0.5;
		frequency *= 2.0;
	}
	return value;
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
	float a  = roughness * roughness;
	float a2 = a * a;
	float NdotH  = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;
	float denom  = NdotH2 * (a2 - 1.0) + 1.0;
	return a2 / (3.14159265 * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;
	return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 subsurfaceScattering(vec3 L, vec3 V, vec3 N, vec3 albedo, float thickness) {
	vec3 scatterDir = L + N * 0.5;
	float VdotS = pow(clamp(dot(V, -scatterDir), 0.0, 1.0), 3.0);
	return albedo * VdotS * thickness;
}

void main() {
	vec4 tex_raw = texture(tex_sampler, v_uv);
	vec3 base_color = v_color * tex_raw.rgb;
	float roughness = clamp(v_mat_data.x, 0.05, 1.0);
	float metalness = v_mat_data.y;

	vec3 N = normalize(v_normal);
	vec3 V = normalize(ubo.eye_pos - v_pos);

	vec3 F0 = mix(vec3(0.04), base_color, metalness);

	vec3 total_lighting = vec3(0.0);

	for (int i = 0; i < 8; i++) {
		float angle = float(i) * 0.785398; // 2*PI / 8
		float radius = 50.0 + float(i) * 20.0;
		vec3 light_offset = vec3(
			cos(angle) * radius,
			sin(angle * 0.7) * 30.0,
			sin(angle) * radius
		);
		vec3 light_p = ubo.light_pos + light_offset;
		vec3 light_c = ubo.light_color * vec3(
			0.5 + 0.5 * sin(angle),
			0.5 + 0.5 * sin(angle + 2.094),
			0.5 + 0.5 * sin(angle + 4.189)
		);
		float intensity = ubo.light_intensity * 0.3;

		vec3 L = normalize(light_p - v_pos);
		vec3 H = normalize(V + L);

		float dist = length(light_p - v_pos);
		float attenuation = intensity / (1.0 + 0.005 * dist * dist);
		vec3 radiance = light_c * attenuation;

		float NDF = distributionGGX(N, H, roughness);
		float G   = geometrySmith(N, V, L, roughness);
		vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

		vec3 numerator = NDF * G * F;
		float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
		vec3 specular = numerator / denominator;

		vec3 kS = F;
		vec3 kD = (vec3(1.0) - kS) * (1.0 - metalness);
		vec3 diffuse = kD * base_color / 3.14159265;

		float NdotL = max(dot(N, L), 0.0);
		total_lighting += (diffuse + specular) * radiance * NdotL;

		total_lighting += subsurfaceScattering(L, V, N, base_color, 0.3) * attenuation * 0.1;
	}

	float detail = fbm(v_pos * 2.0) * 0.15;
	total_lighting += base_color * detail;

	float ao = 1.0;
	for (int i = 0; i < 4; i++) {
		float scale = 0.5 + float(i) * 0.5;
		vec3 sample_pos = v_pos + N * scale * 0.1;
		ao -= (scale * 0.1 - noise3D(sample_pos * 5.0) * scale * 0.1) * 0.5;
	}
	ao = clamp(ao, 0.0, 1.0);

	vec3 ambient = 0.06 * base_color * ao;
	vec3 result = ambient + total_lighting * ao;

	if (v_emissive > 0.5) {
		result += base_color * 2.0;
	}

	float fog_dist = length(ubo.eye_pos - v_pos);
	float fog = exp(-0.002 * fog_dist);
	result = mix(vec3(0.02, 0.02, 0.03), result, fog);

	vec2 center_dist = v_screen_uv - 0.5;
	float vignette = 1.0 - dot(center_dist, center_dist) * 1.2;
	result *= clamp(vignette, 0.0, 1.0);

	result = (result * (2.51 * result + 0.03)) / (result * (2.43 * result + 0.59) + 0.14);

	f_color = vec4(clamp(result, 0.0, 1.0), tex_raw.a);
}
