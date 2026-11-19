{
  description = "Out-of-core GGUF layer-streaming inference engine and Mesh-LLM plugin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/0bb2eedc20e4739f29e8045034f2828ad7054f8f";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/0693dbfb915f33233818373b5f6969396658d02c";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      llama-cpp-src,
    }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);

      devShells = forAllSystems (
        pkgs:
        let
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustc
                cargo
                rustfmt
                clippy
                cmake
                ninja
                pkg-config
                protobuf
                just
                curl
                jq
              ]
              ++ lib.optionals isLinux [
                vulkan-headers
                vulkan-loader
                shaderc
              ];

            LLAMA_CPP_DIR = "${llama-cpp-src}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            VULKAN_SDK = lib.optionalString isLinux "${pkgs.vulkan-headers}";
          };
        }
      );
    };
}
