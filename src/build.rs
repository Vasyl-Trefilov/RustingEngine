pub fn main() {
    println!("cargo:rerun-if-changed=src/shaders/physics.comp");
}