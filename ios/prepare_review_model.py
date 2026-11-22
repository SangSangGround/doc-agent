"""Prepare the offline Qwen and Whisper models before building the iPhone app."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

SOURCES = Path(__file__).resolve().parent / "DocAgentCore/Sources"
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


def prepare(directory, base_url, files):
    directory.mkdir(exist_ok=True, parents=True)
    for name, expected in files.items():
        target = directory / name
        if target.is_file() and digest(target) == expected:
            continue
        print(f"Downloading {name}", flush=True)
        with tempfile.TemporaryDirectory(dir=SOURCES.parent) as staging:
            download = Path(staging) / name
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "3", "--output", str(download), f"{base_url}/{name}?download=true"],
                           check=True)
            if digest(download) != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            download.replace(target)

def main():
    prepare(SOURCES / "DocAgentCore/ReviewModel", BASE, FILES)
    prepare(SOURCES / "QuestionSpeech/SpeechModel",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1",
            {"ggml-base.bin": "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"})
    prepare(SOURCES / "QuestionSpeech/SpeechModel",
            "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b",
            {"ggml-silero-v6.2.0.bin": "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"})
    print("Offline review and speech models are ready.")


if __name__ == "__main__":
    main()
