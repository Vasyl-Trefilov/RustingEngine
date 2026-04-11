#version 450

// FRAGMENT SHADER - PBR (Physically Based Rendering)

// This shader implements PBR lighting with:
// - Cook-Torrance BRDF (specular)
// - Lambertian diffuse
// - Image-based lighting approximation
// - Fog and vignette post-processing
// - Tone mapping (ACES approximation)

// INPUT VARYINGS (from vertex shader)


layout(location = 0) in vec3 v_color;         // Base object color RGB
layout(location = 1) in vec3 v_normal;        // World-space normal
layout(location = 2) in vec3 v_pos;           // World-space position
layout(location = 3) in vec4 v_mat_data;    // x = roughness, y = metalness
layout(location = 4) in vec2 v_uv;          // Texture coordinates
layout(location = 5) in float v_emissive;    // Emissive intensity (0 = normal, >0 = glowing)
layout(location = 6) in vec2 v_screen_uv;   // Screen-space UV for post-processing


// UNIFORM BUFFER - Camera and light data


layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;           // View matrix
    mat4 proj;           // Projection matrix
    vec3 eye_pos;       // Camera position (unused here, passed to main)
    float pad1;         // Padding
    vec3 light_pos;     // Point light position
    float pad2;          // Padding
    vec3 light_color;   // Point light RGB color
    float light_intensity;  // Point light brightness
} ubo;

// TEXTURE SAMPLER


layout(set = 0, binding = 1) uniform sampler2D tex_sampler;

// OUTPUT COLOR


layout(location = 0) out vec4 f_color;

// MAIN ENTRY POINT


void main() {
    // EMISSIVE MATERIAL (self-illuminating)
    
    // If the material has emissive enabled, show it as glowing
    if (v_emissive > 0.5) {
        // Boost the color for glow effect
        vec3 result = v_color * 1.5;
        // Apply tone mapping
        result = vec3(1.0) - exp(-result * 1.2);
        f_color = vec4(result, 1.0);
        return;
    }

    // Sample the texture (contains albedo in RGB, alpha in A)
    vec4 tex_raw = texture(tex_sampler, v_uv);
    
    // Multiply base color by texture
    vec3 base_color = v_color * tex_raw.rgb;

    // MATERIAL PROPERTIES
    
    // Clamp roughness to avoid division by zero and overly shiny surfaces
    float roughness = clamp(v_mat_data.x, 0.05, 1.0);
    float metalness = v_mat_data.y;

    // VECTORS FOR LIGHTING

    
    vec3 N = normalize(v_normal);              // Surface normal
    vec3 V = normalize(ubo.eye_pos - v_pos);  // View direction
    vec3 L = normalize(ubo.light_pos - v_pos); // Light direction
    vec3 H = normalize(V + L);               // Half direction

    // LIGHT ATTENUATION
    
    // Calculate distance from light to surface
    float dist = length(ubo.light_pos - v_pos);
    
    // Inverse square attenuation with a minimum distance
    float attenuation = ubo.light_intensity / (1.0 + 0.005 * dist * dist);
    
    // Light radiance (color adjusted by attenuation)
    vec3 radiance = ubo.light_color * attenuation;

    // COOK-TORRANCE BRDF

    
    // F0: Surface reflection at zero incidence
    // Non-metals have F0 = 0.04, metals have F0 = base_color
    vec3 f0 = mix(vec3(0.04), base_color, metalness);
    
    // Fresnel-Schlick approximation
    float cosTheta = max(dot(H, V), 0.0);
    vec3 F = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);

    // GEOMETRY TERM (simplified Smith)
    
    float ndotl = max(dot(N, L), 0.0);  // N dot L
    float ndotv = max(dot(N, V), 0.0);  // N dot V

    // DISTRIBUTION TERM (GGX)
    
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float ndoth = max(dot(N, H), 0.0);
    float denom = (ndoth * ndoth * (alpha2 - 1.0) + 1.0);
    float D = alpha2 / (3.14159 * denom * denom);

    // Specular = D * F / (4 * NdotV * NdotL)
    vec3 specular = (D * F) / (4.0 * ndotv * ndotl + 0.001);

    // DIFFUSE (Lambertian)
    
    // Energy that doesn't go to specular
    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metalness);
    
    // Lambertian diffuse
    vec3 diffuse = kD * base_color / 3.14159;

    // FINAL LIGHTING
    
    // Ambient light (never goes below this)
    vec3 ambient = 0.05 * base_color;
    
    // Direct light contribution
    vec3 lighting = (diffuse + specular) * radiance * ndotl;

    // Combine ambient and direct lighting
    vec3 result = ambient + lighting;

    // FOG (distance-based atmospheric scattering)
    
    float fog_dist = length(ubo.eye_pos - v_pos);
    float fog = exp(-0.002 * fog_dist);
    // Mix between fog color and surface color based on distance
    result = mix(vec3(0.02, 0.02, 0.03), result, fog);

    // VIGNETTE (darker at screen edges)
    
    vec2 center_dist = v_screen_uv - 0.5;
    float vignette = 1.0 - dot(center_dist, center_dist) * 1.2;
    result *= clamp(vignette, 0.0, 1.0);

    // TONE MAPPING (ACES approximation - cinematic look)
    
    result = vec3(1.0) - exp(-result * 1.2); 

    // Output final color with alpha from texture
    f_color = vec4(result, tex_raw.a);
}