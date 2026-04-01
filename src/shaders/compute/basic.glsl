#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct InstanceData {
	mat4 model; // [3].xyz = position, [0..2] = rotation 0=x, 1=y, 2=z
	vec4 color; // color.rgb
	vec4 mat_props; // x is metallic, y is roughness, z is ambient occlusion, w is unused
	vec4 velocity; // x, y, z is velocity. w is collision radius
	vec4 physic;  // x is collision type ( type, if < 0.5 => Box, > 0.5 => Sphere), y is mass, z is gravity power, w is unused
	vec4 rotation; // rotate speed
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
    return mat3(
        0.0, v.z, -v.y,
        -v.z, 0.0, v.x,
        v.y, -v.x, 0.0
    );
}
void main() {
	uint i = gl_GlobalInvocationID.x + pc.offset;
	if (i >= pc.offset + pc.count) return;

	write_buf.data[i].model[3].y += 0.0;

	InstanceData me = read_buf.data[i];
	if (me.physic.y <= 0.0) {
		write_buf.data[i] = me;
		return;
	}

    vec3 scale = vec3(
        length(me.model[0].xyz),
        length(me.model[1].xyz),
        length(me.model[2].xyz)
    );
    
    float bound_y = (me.physic.x > 0.5) ? scale.x : (scale.y * 0.5);

    mat3 rotA = mat3(
        me.model[0].xyz / scale.x,
        me.model[1].xyz / scale.y,
        me.model[2].xyz / scale.z
    );
    
    if( length(me.rotation.xyz) > 0.001 ) {
        rotA += (skew(me.rotation.xyz) * rotA) * pc.dt;
        
        vec3 c0 = normalize(rotA[0]);
        vec3 c1 = normalize(rotA[1] - dot(c0, rotA[1]) * c0);
        vec3 c2 = cross(c0, c1);
        rotA = mat3(c0, c1, c2);
    }

    me.model[0] = vec4(rotA[0] * scale.x, 0.0);
    me.model[1] = vec4(rotA[1] * scale.y, 0.0);
    me.model[2] = vec4(rotA[2] * scale.z, 0.0);

    me.velocity.y -= 9.81 * me.physic.z * pc.dt; 
    me.model[3].xyz += me.velocity.xyz * pc.dt;
    
    if(me.model[3].y - bound_y < 0.0){
        me.model[3].y = bound_y;
        me.velocity.y *= -0.5;
    }

    write_buf.data[i] = me;
}
