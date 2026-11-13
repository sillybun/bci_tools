from __future__ import annotations

from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_DIR / "assets"
SOURCE = ASSETS_DIR / "app-icon-source-v3.png"


def main() -> None:
    image = Image.open(SOURCE).convert("RGBA")
    image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
    image.resize((512, 512), Image.Resampling.LANCZOS).save(ASSETS_DIR / "app-icon.png")
    image.save(
        ASSETS_DIR / "app-icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    image.save(
        ASSETS_DIR / "app-icon.icns",
        sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
    )


if __name__ == "__main__":
    main()
