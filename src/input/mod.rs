use winit::event::VirtualKeyCode;
use std::collections::HashSet;

// ! MOUSE 
#[derive(Clone, Copy, Debug)]
pub struct MouseState {
    position: (f32, f32),      // Normalized coordinates (-1 to 1), so its like a vulkan type
    pixel_position: (f32, f32), // Pixel coordinates, like from 0 to 1900
    left_clicked: bool,
    right_clicked: bool,
    left_pressed: bool,
    right_pressed: bool,
    inside_window: bool,
}

impl Default for MouseState {
    // #[inline] // ? I founded this in library, what does this thing? So I read a bit about this, this is for performance, but its making a compiler time longer, so I will leave it commented, maybe after some time I will uncomment it
    fn default() -> Self {
        Self {
            position: (0.0, 0.0),
            pixel_position: (0.0, 0.0),
            left_clicked: false,
            right_clicked: false,
            left_pressed: false,
            right_pressed: false,
            inside_window: true,
        }
    }
}

pub struct InputState {
    pub keys_pressed: HashSet<VirtualKeyCode>,
    pub mouse_delta: [f32; 2],
    pub mouse_captured: bool,
    pub last_mouse_pos: [f32; 2],
    pub is_mouse_dragging: bool,
}