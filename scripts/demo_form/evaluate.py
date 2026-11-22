"""Run real macOS Vision OCR against local, deterministic photographed-page variants.

Example: venv/bin/python scripts/demo_form/evaluate.py IMG_2510.heic --other-pages document{1,2,3}.jpg --output /tmp/demo-form-metrics.json
Requires Xcode, Python OpenCV and libheif's heif-convert. Only measurements are written to --output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--other-pages", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    core = root / "ios/DocAgentCore/Sources/DocAgentCore"
    with tempfile.TemporaryDirectory(prefix="docagent-form-") as directory:
        work = Path(directory)
        decoded = work / "source.png"
        # Use libheif to decode pixels independently of the Apple's image decoder used by the app.
        subprocess.run(["heif-convert", str(args.photo), str(decoded)], check=True, stdout=subprocess.DEVNULL)
        original = cv2.imread(str(decoded))
        if original is None or original.shape[:2] != (3156, 2344):
            raise ValueError("The expected reference is IMG_2510.heic (2344 × 3156).")
        h, w = 1800, 1340
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        manifest = []

        def add(name, pixels, positive, matrix=None):
            path = work / f"{name}.jpg"
            cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append(dict(name=name, path=str(path), positive=positive,
                                 matrix=None if matrix is None else matrix.flatten().tolist(),
                                 height=pixels.shape[0], width=pixels.shape[1]))

        manifest.append(dict(name="source_decoded", path=str(decoded), positive=True, matrix=np.eye(3).flatten().tolist(),
                             width=2344, height=3156))

        def warp(name, corners, source=original, shadow=False):
            pose = cv2.getPerspectiveTransform(unit, np.float32(corners))
            transform = np.diag([w,h,1.]) @ pose @ np.diag([1/source.shape[1],1/source.shape[0],1.])
            im = cv2.warpPerspective(source, transform, (w,h), borderValue=(69, 91, 108))
            if shadow:
                illumination = (0.58 + 0.42*np.linspace(0,1,w))[None, :, None]
                im = np.uint8(im * illumination)
            add(name, im, True, pose)

        warp("resized_margin", [[.10,.09],[.89,.09],[.89,.9],[.10,.9]])
        warp("left_perspective", [[.08,.12],[.92,.07],[.86,.89],[.17,.94]])
        warp("right_perspective", [[.13,.07],[.85,.13],[.94,.92],[.07,.87]])
        warp("mild_shadow", [[.09,.08],[.91,.09],[.90,.93],[.10,.92]], shadow=True)
        inked = original.copy()
        for x, y in [(1684,1674), (1983,2292)]:
            cv2.line(inked, (x-10,y-7), (x-1,y+9), (32,38,42), 5)
            cv2.line(inked, (x-1,y+9), (x+16,y-15), (32,38,42), 5)
        cv2.polylines(inked, [np.int32([[1680,2900],[1720,2878],[1700,2912],[1780,2886],[1840,2906]])],
                      False, (30,30,30), 5)
        warp("fresh_ink", [[.09,.08],[.90,.09],[.90,.93],[.09,.92]], source=inked)
        # Challenge samples reserved after tuning the matcher with the original and the five variants above.
        warp("heldout_offset_keystone", [[.055,.081],[.898,.167],[.792,.92],[.128,.845]], shadow=True)
        blurred = cv2.GaussianBlur(original, (9,9), 2.1)
        warp("heldout_soft_focus", [[.194,.11],[.897,.064],[.929,.832],[.138,.908]], source=blurred)

        full = cv2.resize(original, (w,h), interpolation=cv2.INTER_AREA)
        edited = full.copy()
        cv2.rectangle(edited,(int(w*.344),int(h*.076)),(int(w*.427),int(h*.098)), (195,195,195),-1)
        cv2.putText(edited,"2025.1.1.>",(int(w*.346),int(h*.094)),cv2.FONT_HERSHEY_SIMPLEX,
                    w/2450.,(42,42,42),1,cv2.LINE_AA)
        add("changed_revision", edited, False)
        altered = full.copy()
        cv2.rectangle(altered, (int(w*.426),int(h*.475)), (int(w*.481),int(h*.493)), (194,194,194),-1)
        cv2.putText(altered,"5",(int(w*.432),int(h*.489)),cv2.FONT_HERSHEY_SIMPLEX,
                    w/2250.,(30,30,30),1,cv2.LINE_AA)
        add("changed_retention", altered, False)
        add("header_only", full[:int(h*.24)], False)
        add("no_header", full[int(h*.25):], False)
        for index, other in enumerate(args.other_pages,1):
            other_im = cv2.imread(str(other))
            if other_im is None:
                raise ValueError(f"Unable to decode {other}")
            # The supplied teammate fixture photos store pixel data rotated counterclockwise.
            if other_im.shape[1] > other_im.shape[0]:
                other_im = cv2.rotate(other_im, cv2.ROTATE_90_CLOCKWISE)
            add(f"other_form_{index}", other_im, False)
            if index == 1:
                impostor = cv2.resize(other_im, (w,h))
                impostor[:int(h*.148)] = full[:int(h*.148)]
                add("same_exact_header_different_body", impostor, False)

        manifest_file = work / "cases.json"
        manifest_file.write_text(json.dumps(manifest))
        executable = work / "measure"
        subprocess.run(["xcrun","swiftc","-O","-module-cache-path",str(work/"modules"),
                        *(str(core / f"{file}.swift") for file in
                          ["PhotoReader","WritingGuidance","DemoConsentReference","DemoConsentForm"]),
                        str(Path(__file__).with_name("measure.swift")),"-o",str(executable)],check=True)
        subprocess.run([str(executable),str(manifest_file),str(args.photo),str(args.output)],check=True)


if __name__ == "__main__":
    main()
