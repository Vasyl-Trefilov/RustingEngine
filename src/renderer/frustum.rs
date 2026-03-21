//! CPU-side frustum extraction from view-projection (column-vector convention matching the shaders).

use nalgebra::{Matrix4, Vector4};

// * Build a column-major world matrix from the same `[[f32; 4]; 4]` layout used by the renderer.
#[inline]
pub fn mat4_from_row_array(m: &[[f32; 4]; 4]) -> Matrix4<f32> {
    Matrix4::new(
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3],
    )
}

// * `clip = proj * view * world_pos` — same order as the vertex shader.
#[inline]
pub fn view_proj_matrix(view: &[[f32; 4]; 4], proj: &[[f32; 4]; 4]) -> Matrix4<f32> {
    mat4_from_row_array(proj) * mat4_from_row_array(view)
}

// * Six(Seven) inward-facing clip-space frustum planes in world space (normalized).
pub fn frustum_planes(vp: &Matrix4<f32>) -> [Vector4<f32>; 6] {
    let r0 = vp.row(0);
    let r1 = vp.row(1);
    let r2 = vp.row(2);
    let r3 = vp.row(3);
    // Row-vector combinations; normalize as column vectors for plane equations.
    let left = normalize_plane((r3 + r0).transpose());
    let right = normalize_plane((r3 - r0).transpose());
    let bottom = normalize_plane((r3 + r1).transpose());
    let top = normalize_plane((r3 - r1).transpose());
    let near = normalize_plane((r3 + r2).transpose());
    let far = normalize_plane((r3 - r2).transpose());
    [left, right, bottom, top, near, far]
}

#[inline]
fn normalize_plane(p: Vector4<f32>) -> Vector4<f32> {
    let l = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
    if l < 1e-8 {
        return p;
    }
    p / l
}

// * Conservative sphere vs frustum (all planes must pass).
#[inline]
pub fn sphere_visible(planes: &[Vector4<f32>; 6], center: [f32; 3], radius: f32) -> bool {
    let c = Vector4::new(center[0], center[1], center[2], 1.0);
    for p in planes {
        if p.dot(&c) < -radius {
            return false;
        }
    }
    true
}
