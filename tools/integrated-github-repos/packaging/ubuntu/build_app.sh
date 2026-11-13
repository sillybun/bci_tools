#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

BUILD_VENV="${BUILD_VENV:-$PROJECT_DIR/.packaging-venv}"
if [[ ! -x "$BUILD_VENV/bin/python" ]]; then
  python3 -m venv "$BUILD_VENV"
fi
BUILD_PYTHON="$BUILD_VENV/bin/python"
"$BUILD_PYTHON" -m pip install '.[dev]'
"$BUILD_PYTHON" packaging/generate_icons.py
"$BUILD_PYTHON" -m PyInstaller --noconfirm --clean --windowed \
  --name "integrated-github-repos" \
  --icon assets/app-icon.png \
  --add-data "assets/app-icon.png:assets" \
  --paths src \
  src/integrated_github_repos/app.py

echo "Built: $PROJECT_DIR/dist/integrated-github-repos/integrated-github-repos"
