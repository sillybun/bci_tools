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
  --name "Integrated GitHub Repos" \
  --osx-bundle-identifier "io.github.bertie97.integrated-github-repos" \
  --icon assets/app-icon.icns \
  --add-data "assets/app-icon.png:assets" \
  --paths src \
  src/integrated_github_repos/app.py

CODESIGN_IDENTITY_VALUE="${CODESIGN_IDENTITY:-}"
if [[ -n "$CODESIGN_IDENTITY_VALUE" ]]; then
  codesign --force --deep --options runtime --sign "$CODESIGN_IDENTITY_VALUE" \
    "dist/Integrated GitHub Repos.app"
else
  codesign --force --deep --sign - "dist/Integrated GitHub Repos.app"
fi
codesign --verify --deep --strict "dist/Integrated GitHub Repos.app"
echo "Built: $PROJECT_DIR/dist/Integrated GitHub Repos.app"
