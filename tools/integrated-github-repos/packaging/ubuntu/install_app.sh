#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUNDLE="$PROJECT_DIR/dist/integrated-github-repos"
INSTALL_DIR="$HOME/.local/opt/integrated-github-repos"
APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

if [[ ! -x "$BUNDLE/integrated-github-repos" ]]; then
  echo "Run packaging/ubuntu/build_app.sh first." >&2
  exit 1
fi

# Qt's xcb plugin needs this library, which is absent on some Ubuntu installs.
# Download the Ubuntu package locally rather than requiring a system-wide install.
DEPS_DIR="$(mktemp -d)"
trap 'rm -rf "$DEPS_DIR"' EXIT
(
  cd "$DEPS_DIR"
  apt-get download libxcb-cursor0
  dpkg-deb -x ./*.deb extracted
)
LIBRARY="$(find "$DEPS_DIR/extracted" -name 'libxcb-cursor.so.0' -print -quit)"
[[ -n "$LIBRARY" ]] || { echo "libxcb-cursor.so.0 not found" >&2; exit 1; }

mkdir -p "$INSTALL_DIR" "$APPLICATIONS_DIR"
cp -a "$BUNDLE/." "$INSTALL_DIR/"
# PyInstaller resolves bundled libraries from _internal, not the executable directory.
cp -L "$LIBRARY" "$INSTALL_DIR/_internal/libxcb-cursor.so.0"
mkdir -p "$INSTALL_DIR/licenses/libxcb-cursor0"
cp "$DEPS_DIR/extracted/usr/share/doc/libxcb-cursor0/copyright" \
  "$INSTALL_DIR/licenses/libxcb-cursor0/copyright"

cat > "$APPLICATIONS_DIR/integrated-github-repos.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Integrated GitHub Repos
Comment=Local Git repository desktop client
Exec="$INSTALL_DIR/integrated-github-repos"
Icon=$INSTALL_DIR/_internal/assets/app-icon.png
Terminal=false
Categories=Development;
StartupWMClass=integrated-github-repos
EOF
if command -v desktop-file-validate >/dev/null; then
  desktop-file-validate "$APPLICATIONS_DIR/integrated-github-repos.desktop"
fi
if command -v update-desktop-database >/dev/null; then
  update-desktop-database "$APPLICATIONS_DIR"
fi
echo "Installed: $INSTALL_DIR/integrated-github-repos"
