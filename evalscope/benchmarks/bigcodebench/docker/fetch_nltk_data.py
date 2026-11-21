"""Installs the NLTK data packages of a lock file (install directory, URL, SHA-256 per line) into DEST: each archive is
downloaded, checked against its SHA-256 and unpacked into DEST/<install directory>, refusing members with absolute
paths or parent references. Runs at image build time (Dockerfile).

    fetch_nltk_data.py LOCK DEST
"""

import hashlib
import io
import pathlib
import sys
import urllib.request
import zipfile

lock, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for line in lock.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    directory, url, sha256 = line.split("\t")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"{url}: SHA-256 {digest}, expected {sha256}")
    archive = zipfile.ZipFile(io.BytesIO(data))
    for name in archive.namelist():
        if name.startswith("/") or ".." in pathlib.PurePosixPath(name).parts:
            sys.exit(f"{url}: unsafe member {name}")
    target = dest / directory
    target.mkdir(parents=True, exist_ok=True)
    archive.extractall(target)
    print(f"installed {url.rsplit('/', 1)[1]} into {target}")
