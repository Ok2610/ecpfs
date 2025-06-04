// build.rs
fn main() {
    println!(
        "cargo:rustc-link-search=native={}",
        r"C:\tools\vcpkg\installed\x64-windows-static-md\lib"
    );

    println!("cargo:rustc-link-lib=static=openblas");
    println!("cargo:rerun-if-changed=build.rs");
}
