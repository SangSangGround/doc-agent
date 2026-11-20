"""Prepare the two offline Qwen vision-language model files before building the iPhone app."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

MODELS = Path(__file__).resolve().parent / "DocAgentCore/Sources/DocAgentCore/ReviewModel"
BASE = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
FILES = {
    "Qwen3.5-0.8B-Q4_K_M.gguf": "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517",
    "mmproj-F16.gguf": "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453",
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    MODELS.mkdir(exist_ok=True, parents=True)
    for name, expected in FILES.items():
        target = MODELS / name
        if target.is_file() and digest(target) == expected:
            continue
        print(f"Downloading {name}", flush=True)
        with tempfile.TemporaryDirectory(dir=MODELS.parents[3]) as staging:
            download = Path(staging) / name
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "3", "--output", str(download), f"{BASE}/{name}?download=true"],
                           check=True)
            if digest(download) != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            download.replace(target)
    print("Offline review model is ready.")


if __name__ == "__main__":
    main()
