#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel packaging ninja
.venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install mamba-ssm==2.3.1 --no-build-isolation
rustup toolchain install 1.85.1 --profile minimal
.venv/bin/python -B reproduce.py check
