fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Vendor protoc so the build needs no system `protoc` install.
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    tonic_build::configure()
        .build_client(false)
        .compile_protos(
            &["../../proto/memlake/v1/memlake.proto"],
            &["../../proto"],
        )?;
    Ok(())
}
