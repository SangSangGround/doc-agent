"""Replay known projective transforms against the production iOS ORB implementation on macOS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageOps


def load(path, directory, longest):
    if path.suffix.lower() in {".heic", ".heif"}:
        decoded = directory / "source-converted.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(decoded)],
                       check=True, stdout=subprocess.DEVNULL)
        path = decoded
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((longest, longest))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def prepare(reference, negatives, directory):
    ref = load(reference, directory, 1600)
    cv2.imwrite(str(directory / "reference.png"), ref)
    h, w = 960, 720
    src = np.float32([[0, 0], [ref.shape[1], 0], [ref.shape[1], ref.shape[0]], [0, ref.shape[0]]])
    # Centers of the four annotated boxes and the applicant signature area on the supplied HEIC.
    anchors = np.float32([[.718645, .53026], [.841935, .53074], [.722055, .72592],
                          [.845, .726], [.77, .9185]])
    frames = []
    quads = {
        "upright": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "surround": [(.12, .08), (.88, .08), (.88, .92), (.12, .92)],
        "perspective-left": [(.18, .09), (.91, .18), (.83, .95), (.07, .82)],
        "perspective-right": [(.10, .20), (.78, .08), (.95, .85), (.18, .96)],
        "quarter-turn": [(.92, .18), (.92, .8), (.08, .8), (.08, .18)],
        "partial": [(-.40, -.57), (1.11, -.48), (1.13, 1.06), (-.35, 1.12)],
        "hand-shaped-occlusion": [(.05, .06), (.96, .04), (.92, .96), (.04, .94)],
        "dim": [(.04, .02), (.98, .08), (.97, .99), (.04, .96)],
        "mirrored": [(1, 0), (0, 0), (0, 1), (1, 1)],
    }
    for name, corners in quads.items():
        matrix = cv2.getPerspectiveTransform(src, np.float32(corners)*[np.float32(w),np.float32(h)])
        live = cv2.warpPerspective(ref, matrix, (w,h), borderValue=(44,50,67))
        if name == "hand-shaped-occlusion":
            cv2.ellipse(live, (270, 710), (74, 150), 15, 0, 360, (121, 160, 206), -1)
        if name == "dim":
            live = np.uint8(live * 0.64 + 14)
        xy = cv2.perspectiveTransform((anchors * [np.float32(ref.shape[1]),np.float32(ref.shape[0])]).reshape(-1,1,2), matrix).reshape(-1,2)
        points = [[float(x),float(y),float(u/w),float(v/h)] for (x,y),(u,v) in zip(anchors, xy)]
        filename = name + ".jpg"
        cv2.imwrite(str(directory / filename), live, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(dict(file=filename, expected=name != "mirrored", landmarks=points))
    cv2.imwrite(str(directory / "blank.png"), np.full((h,w,3), 215, dtype=np.uint8))
    frames.append(dict(file="blank.png", expected=False, landmarks=[]))
    # Near-negative: retain original header/footer/table dimensions, but replace
    # the central printed rows. Header-only feature registration is insufficient.
    lookalike = cv2.resize(ref, (w,h))
    top, bottom = h//6, h*5//6
    lookalike[top:bottom] = cv2.flip(lookalike[top:bottom], 0)
    cv2.imwrite(str(directory / "same-header-changed-body.jpg"), lookalike)
    frames.append(dict(file="same-header-changed-body.jpg", expected=False, landmarks=[]))
    for i, path in enumerate(negatives):
        filename = f"other-{i+1}.jpg"
        cv2.imwrite(str(directory / filename), load(path, directory, 1080))
        frames.append(dict(file=filename, expected=False, landmarks=[]))
    (directory / "frames.json").write_text(json.dumps(dict(reference="reference.png",frames=frames)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--negative", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path, help="Output JSON with per-frame registration/latency/reprojection errors")
    parser.add_argument("--scratch-path", type=Path)
    parser.add_argument("--cache-path", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as folder:
        directory = Path(folder)
        prepare(args.reference, args.negative, directory)
        run = ["swift", "test", "--package-path", str(root / "ios/DocAgentCore"), "--filter", "WritingGuidanceTests.testRegistrationReplay"]
        if args.scratch_path:
            run += ["--scratch-path", str(args.scratch_path)]
        if args.cache_path:
            run += ["--cache-path", str(args.cache_path)]
        env = dict(os.environ, DOCAGENT_WRITING_REPLAY=str(directory))
        process = subprocess.run(run, env=env, check=False)
        results = directory / "registration-results.json"
        if results.exists():
            args.output.write_bytes(results.read_bytes())
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
