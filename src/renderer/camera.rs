pub struct Camera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

pub fn create_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let v = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/len, v[1]/len, v[2]/len]
    };
    let s = {
        let v = [
            f[1] * up[2] - f[2] * up[1],
            f[2] * up[0] - f[0] * up[2],
            f[0] * up[1] - f[1] * up[0],
        ];
        let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/len, v[1]/len, v[2]/len]
    };
    let u = [
        s[1] * f[2] - s[2] * f[1],
        s[2] * f[0] - s[0] * f[2],
        s[0] * f[1] - s[1] * f[0],
    ];

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [
            -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]),
            -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]),
            (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]),
            1.0
        ],
    ]
    // ? Why is here so much math, I am tired of math a bit
}
 

pub fn create_projection_matrix(aspect: f32, fov: f32, z_near: f32, z_far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, z_far / (z_far - z_near), 1.0],
        [0.0, 0.0, -(z_far * z_near) / (z_far - z_near), 0.0],
    ]
}