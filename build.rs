use std::{env, path::Path};

const LIB_NAMES: &[&str] = &["libmp3lame.dylib", "libmp3lame.so", "libmp3lame.a"];
const DEFAULT_SEARCH_PATHS: &[&str] = &[
    "/usr/lib",
    "/usr/local/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/opt/homebrew/lib",
    "/opt/local/lib",
];

fn main() {
    println!("cargo:rerun-if-env-changed=LIBMP3LAME_DIR");
    if let Some(dir) = env::var_os("LIBMP3LAME_DIR") {
        println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
        return;
    }

    if let Some(path) = find_existing_path(DEFAULT_SEARCH_PATHS) {
        println!("cargo:rustc-link-search=native={}", path);
    }
}

fn find_existing_path<'a>(candidates: &'a [&'a str]) -> Option<&'a str> {
    for path in candidates {
        if LIB_NAMES
            .iter()
            .any(|name| Path::new(path).join(name).exists())
        {
            return Some(path);
        }
    }
    None
}
