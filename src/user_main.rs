#![allow(dead_code)]

//! This is a test file, where I test the user-friendliness of my library
mod examples;
use crate::examples::physics::main as physics_main;
use crate::examples::rotate::main as rotate_main;
use crate::examples::shaders::main as shaders_main;
use crate::examples::stress_pbr::main as pbr_main;
use crate::examples::stress_unlit::main as unlit_main;

fn main() {
    // Comment out the function you don't want to run, and uncomment the one you do!

    // shaders_main();
    // physics_main();

    // 10,000 cubes stress testing
    // unlit_main(); 
    pbr_main();

    // rotate test
    // rotate_main();
}
