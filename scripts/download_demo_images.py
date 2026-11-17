"""Download six real public source photos solely for a functional smoke gallery."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from PIL import Image

URLS = {
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "cats.png": "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png",
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    "football.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
    "dog1.jpg": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg",
    "dog2.jpg": "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/demo")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    report = []
    for name, url in URLS.items():
        path = root / name
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        with Image.open(path) as image:
            image.verify()
        report.append({"name": name, "source": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    target = Path("reports/demo_images.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated {len(report)} real photographs in {root}")

