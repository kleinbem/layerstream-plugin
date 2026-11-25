set shell := ["bash", "-euo", "pipefail", "-c"]

# List available recipes (all task logic lives in pure-Rust `xtask/`)
default:
    @just --list

check-versions:
    cargo xtask check-versions

check-loc:
    cargo xtask check-loc

preflight:
    cargo xtask preflight

fmt:
    cargo xtask fmt

test-rust:
    cargo xtask test-rust

coverage:
    cargo xtask coverage

test-python-fast:
    cargo xtask test-python-fast

coverage-python:
    cargo xtask coverage-python

coverage-node:
    cargo xtask coverage-node

lint-cpp:
    cargo xtask lint-cpp

coverage-cpp:
    cargo xtask coverage-cpp

gpu-smoke backend="vulkan":
    cargo xtask gpu-smoke {{backend}}

bump version:
    cargo xtask bump {{version}}
