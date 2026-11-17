import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keds.gallery.repository import Gallery

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gallery", required=True, help="Gallery directory containing CURRENT.json")
    args = p.parse_args()
    print(Gallery(args.gallery).validate())

