"""Download and extract the four official MNIST IDX files (standard-library only)."""
import argparse
import gzip
import urllib.request
from pathlib import Path

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
)
FILENAMES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)

    for filename in FILENAMES:
        destination = args.data / filename.removesuffix(".gz")
        if destination.exists():
            print(f"Already exists: {destination}")
            continue
        errors = []
        for base in MIRRORS:
            try:
                url = f"{base}/{filename}"
                print(f"Downloading {url}")
                request = urllib.request.Request(url, headers={"User-Agent": "mnist-from-scratch/1.0"})
                with urllib.request.urlopen(request, timeout=40) as response:
                    compressed = response.read()
                extracted = gzip.decompress(compressed)
                # Validate the IDX header magic before writing the file.
                expected_magic = 2051 if "images" in filename else 2049
                if int.from_bytes(extracted[:4], "big") != expected_magic:
                    raise ValueError(f"Invalid MNIST IDX header in {filename}")
                destination.write_bytes(extracted)
                print(f"Saved {destination} ({len(extracted):,} bytes)")
                break
            except Exception as exc:
                errors.append(f"{base}: {exc}")
        else:
            raise RuntimeError(f"Failed to download {filename}\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
