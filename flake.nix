{
  description = "Out-of-core GGUF layer-streaming inference engine and Mesh-LLM plugin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/0bb2eedc20e4739f29e8045034f2828ad7054f8f";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/7d9b931df9bb7c689f067a7fed23e7f5253951f6";
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
        "aarch64-darwin"
      ];
      forAllSystems =
        f:
        lib.genAttrs systems (
          system:
          f (
            import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
              }
              // lib.optionalAttrs (system == "aarch64-linux") {
                cudaCapabilities = [ "8.7" ];
              };
            }
          )
        );
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);

      devShells = forAllSystems (
        pkgs:
        let
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
          isAarch64 = pkgs.stdenv.hostPlatform.isAarch64;
          commonPackages =
            with pkgs;
            [
              rustc
              cargo
              rustfmt
              clippy
              rust-analyzer
              clang-tools
              sccache
              cargo-nextest
              cargo-audit
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
          cudaPkgs = pkgs.cudaPackages;
          cudaMerged =
            if isLinux then
              pkgs.symlinkJoin {
                name = "layerstream-cuda-toolkit";
                paths = [
                  cudaPkgs.cuda_nvcc
                  (lib.getDev cudaPkgs.cuda_cudart)
                  (lib.getLib cudaPkgs.cuda_cudart)
                  (lib.getDev cudaPkgs.libcublas)
                  (lib.getLib cudaPkgs.libcublas)
                  (lib.getDev cudaPkgs.cuda_nvml_dev)
                  cudaPkgs.cccl
                ];
              }
            else
              null;
        in
        {
          default = pkgs.mkShell {
            packages = commonPackages;

            LLAMA_CPP_DIR = "${llama-cpp-src}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            VULKAN_SDK = lib.optionalString isLinux "${pkgs.vulkan-headers}";
          };

          python = pkgs.mkShell {
            packages =
              commonPackages
              ++ (with pkgs; [
                (python3.withPackages (
                  ps: with ps; [
                    pytest
                    pytest-asyncio
                    pytest-timeout
                    hypothesis
                    pip
                    setuptools
                    wheel
                  ]
                ))
                maturin
                ruff
                pyright
                uv
                pip-audit
              ]);

            LLAMA_CPP_DIR = "${llama-cpp-src}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            VULKAN_SDK = lib.optionalString isLinux "${pkgs.vulkan-headers}";
          };

          node = pkgs.mkShell {
            packages =
              commonPackages
              ++ (with pkgs; [
                nodejs_24
                typescript
                biome
              ]);

            LLAMA_CPP_DIR = "${llama-cpp-src}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            VULKAN_SDK = lib.optionalString isLinux "${pkgs.vulkan-headers}";
          };
        }
        // lib.optionalAttrs isLinux {
          cuda = pkgs.mkShell.override { stdenv = cudaPkgs.backendStdenv; } {
            packages = commonPackages ++ [
              cudaPkgs.cuda_nvcc
              cudaPkgs.cuda_cudart
              cudaPkgs.libcublas
              cudaPkgs.cuda_nvml_dev
              cudaPkgs.cccl
            ];

            LLAMA_CPP_DIR = "${llama-cpp-src}";
            CUDA_PATH = "${cudaMerged}";
            CUDAToolkit_ROOT = "${cudaMerged}";
            CMAKE_CUDA_COMPILER = "${cudaPkgs.cuda_nvcc}/bin/nvcc";
            CMAKE_CUDA_HOST_COMPILER = "${cudaPkgs.backendStdenv.cc}/bin/g++";
            CMAKE_CUDA_ARCHITECTURES = if isAarch64 then "87" else "89;120a";
            LD_LIBRARY_PATH =
              lib.makeLibraryPath [
                (lib.getLib cudaPkgs.cuda_cudart)
                (lib.getLib cudaPkgs.libcublas)
              ]
              + ":/run/opengl-driver/lib";
          };
        }
      );
    };
}
