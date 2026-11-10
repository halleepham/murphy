"""Download the prebuilt data from this repository's GitHub release into data/.

  python scripts/download_data.py           app data: everything the app reads
  python scripts/download_data.py --graph   also the data for loading the knowledge graph

Each zip is checked against its SHA-256 before it is unpacked. Standard library only.
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/halleepham/murphy/releases/download/v1.0"
FILES = {  # name: SHA-256
    "murphy-app-data.zip": "b2e2d8df5481048d512527d0f7857463b86f564553219ed5442d49c9d47197e7",
    "murphy-graph-data.zip": "3b0bcaf890d173745532251ab66e09a8aaedd4da9db2174f74356bf8ddc4cc38",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str) -> None:
    zpath = ROOT / "data" / name
    zpath.parent.mkdir(exist_ok=True)
    if not (zpath.exists() and sha256(zpath) == FILES[name]):
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(f"{RELEASE}/{name}", zpath)
    got = sha256(zpath)
    if got != FILES[name]:
        raise SystemExit(f"{name}: SHA-256 {got} does not match the published {FILES[name]}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(ROOT)  # paths inside the zip start with data/
    zpath.unlink()
    print(f"{name}: checked and unpacked", flush=True)


if __name__ == "__main__":
    fetch("murphy-app-data.zip")
    if "--graph" in sys.argv[1:]:
        fetch("murphy-graph-data.zip")
