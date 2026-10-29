"""Install rclone from its official release, pinned by version and sha256 (docker build step).

Usage: python3 install_rclone.py <version> <arch> <sha256> [<dest>]

Debian's rclone package is built with an old Go whose standard library carries known CVEs; the
upstream release is built with a current one. The download is refused unless its sha256 equals
the pinned value from the release's SHA256SUMS.
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
import urllib.request
import zipfile


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    version, arch, want = argv[0], argv[1], argv[2].lower()
    dest = argv[3] if len(argv) == 4 else "/usr/local/bin/rclone"
    name = f"rclone-{version}-linux-{arch}"
    url = f"https://github.com/rclone/rclone/releases/download/{version}/{name}.zip"
    data = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
                data = resp.read()
            break
        except OSError as exc:
            print(f"rclone download failed ({exc}); retry {attempt}/5", file=sys.stderr)
            time.sleep(20)
    if data is None:
        print(f"rclone download failed: {url}", file=sys.stderr)
        return 1
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        print(f"rclone sha256 mismatch for {url}: got {got}, pinned {want}", file=sys.stderr)
        return 1
    blob = zipfile.ZipFile(io.BytesIO(data)).read(f"{name}/rclone")
    with open(dest, "wb") as fh:
        fh.write(blob)
    print(f"installed {name} ({len(blob)} bytes, sha256 of zip verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
