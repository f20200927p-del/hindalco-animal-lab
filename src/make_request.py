"""Create the small JSON request body expected by score.py."""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image


def image_as_jpeg_bytes(image_path):
    """Open an image, shrink its longest side to 512 pixels, and encode JPEG."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Build an Azure scoring request")
    parser.add_argument("--image", required=True, help="Image file to send")
    parser.add_argument("--out", default="sample-request.json", help="JSON output path")
    args = parser.parse_args()

    encoded = base64.b64encode(image_as_jpeg_bytes(args.image)).decode("ascii")
    output = Path(args.out)
    output.write_text(json.dumps({"image": encoded}, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()