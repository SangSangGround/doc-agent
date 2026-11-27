import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1680, 2376  # A4 portrait, approximately 8 px/mm


def save_image(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'Encoding failed: {path}')
    data.tofile(str(path))


def rectify(image, corners):
    # TL, TR, BR, BL; intended for upright portrait documents.
    p = np.asarray(corners, dtype=np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError(f'Invalid document corners: {p}')
    top = p[np.argsort(p[:, 1])[:2]]
    bottom = p[np.argsort(p[:, 1])[2:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    src = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    h, w = image.shape[:2]
    if (src[:, 0] < 0).any() or (src[:, 0] > w-1).any() or (src[:, 1] < 0).any() or (src[:, 1] > h-1).any():
        raise ValueError('Document corners outside image. Retake photo.')
    if not cv2.isContourConvex(src) or cv2.contourArea(src) < 0.05 * w * h:
        raise ValueError('Invalid or very small document area. Retake photo.')
    dst = np.array([[0, 0], [WIDTH-1, 0], [WIDTH-1, HEIGHT-1], [0, HEIGHT-1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Invalid homography.')
    out = cv2.warpPerspective(image, H, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REPLICATE)
    return out, H, src


def illumination_correction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 15)
    corrected = cv2.divide(gray, np.maximum(background, 1), scale=255)
    binary = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12)
    return corrected, binary


def draw_points(image, corners):
    # Downscale preview only: all inference coordinates remain in original pixels.
    h, w = image.shape[:2]
    scale = min(1.0, 1500 / max(h, w))
    preview = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    for i, point in enumerate(points):
        if not np.isfinite(point).all():
            continue
        x, y = np.rint(point*scale).astype(int)
        cv2.circle(preview, (x,y), 7, (0,0,255), -1)
        cv2.putText(preview, str(i+1), (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
    return preview


def added_border(image, fraction):
    h, w = image.shape[:2]
    px, py = round(w*fraction), round(h*fraction)
    # Neutral dark margin; never treat padded pixels as genuine paper.
    out = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_CONSTANT, value=(64,64,64))
    return out, np.array([px,py], dtype=np.float32)


def automatic_scan(image, output):
    from docaligner import DocAligner
    print('Loading DocAligner...', flush=True)
    model = DocAligner(model_type='heatmap', model_cfg='fastvit_sa24')
    attempts = []
    def evaluate(name, corners):
        corners = np.asarray(corners, dtype=np.float32).reshape(-1,2)
        print(f'{name}: {len(corners)} corners\n{corners}', flush=True)
        save_image(output/f'{name}.jpg', draw_points(image, corners))
        row = {'method': name, 'corners': corners.tolist(), 'count': len(corners)}
        attempts.append(row)
        try:
            result = rectify(image, corners)
        except ValueError as e:
            row['error'] = str(e)
            return None
        return result

    # Keep public API as first step. Try margins and a photometric-only variant.
    pad, offset = added_border(image, .12)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.cvtColor(cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray),cv2.COLOR_GRAY2BGR)
    variants = [('original',image,np.zeros(2,np.float32)), ('padded',pad,offset), ('contrast',enhanced,np.zeros(2,np.float32))]
    for name, variant, offset in variants:
        result = evaluate(name, np.asarray(model(variant, do_center_crop=False)).reshape(-1,2)-offset)
        if result is not None:
            return result, attempts, name
    # Public API has a fixed cutoff in the upstream heatmap implementation.
    # Use its own preprocessing/postprocessing with lower cutoffs for PoC only.
    try:
        from docaligner.heatmap_reg.infer import preprocess, postprocess
        for name, variant, offset in variants[:2]:
            info = preprocess(img=variant, img_size_infer=model.detector.img_size_infer, do_center_crop=False)
            heatmap = model.detector.model(**info['input'])['heatmap']
            for threshold in (.20,.15):
                pts = postprocess(preds=heatmap.copy(), imgs_size=info['img_size_ori'], heatmap_threshold=threshold)
                method = f'{name}_threshold_{int(threshold*100)}'
                result = evaluate(method, np.asarray(pts).reshape(-1,2)-offset)
                if result is not None:
                    return result, attempts, method
    except (ImportError, AttributeError, TypeError, KeyError) as e:
        print(f'Optional low-threshold retry unavailable: {e}', flush=True)
        attempts.append({'method':'low_threshold_retry','error':str(e)})
    return None, attempts, None


def manual_corners(image):
    # Developer-only fallback, not an accessible end-user UI.
    h,w = image.shape[:2]
    scale = min(1., 850/h, 1200/w)
    sw,sh = round(w*scale),round(h*scale)
    sx,sy = sw/w,sh/h
    view = cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA)
    points = []
    title = 'Click TL, TR, BR, BL | Enter confirm | R reset | Esc cancel'
    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points)<4:
            points.append((x,y))
    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title,click)
    try:
        while True:
            frame = view.copy()
            for i,(x,y) in enumerate(points):
                cv2.circle(frame,(x,y),5,(0,0,255),-1)
                cv2.putText(frame,str(i+1),(x+7,y-7),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            if len(points)>1:
                cv2.polylines(frame,[np.array(points,np.int32)],len(points)==4,(0,255,0),2)
            cv2.imshow(title,frame)
            key = cv2.waitKey(30)&0xff
            if key in (10,13) and len(points)==4:
                return np.asarray(points,np.float32)/np.array([sx,sy],np.float32)
            if key in (ord('r'),ord('R')):
                points.clear()
            if key==27 or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                raise ValueError('Manual selection canceled.')
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', type=Path, default=BASE/'test_document.jpg')
    parser.add_argument('--manual', action='store_true', help='Select four paper corners manually for developer testing')
    args = parser.parse_args()
    source = args.image.resolve()
    image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read: {source}')
    output = BASE/'scan_output'
    output.mkdir(exist_ok=True)
    # Keep old successful outputs separate from a new run to avoid stale results.
    if any(output.iterdir()):
        output = output / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        output.mkdir()
    print('Output directory:', output, flush=True)
    if args.manual:
        try:
            corners = manual_corners(image)
            save_image(output/'manual_points.jpg', draw_points(image,corners))
            result = rectify(image,corners)
        except ValueError as e:
            print(f'Manual scan failed: {e}', flush=True)
            return 2
        attempts, method = [], 'manual'
    else:
        result, attempts, method = automatic_scan(image, output)
    (output/'scan_status.json').write_text(json.dumps({'status':'candidate' if result is not None else 'failed','method':method,'attempts':attempts},indent=2),encoding='utf-8')
    if result is None:
        print('SCAN FAILED: no valid four-corner candidate. Retake with all corners visible, or use --manual for developer testing.', flush=True)
        return 2
    rectified, H, src = result
    preview = image.copy()
    cv2.polylines(preview, [np.rint(src).astype(np.int32)], True, (0, 255, 0), 3)
    for label, (x, y) in zip(('TL', 'TR', 'BR', 'BL'), np.rint(src).astype(int)):
        cv2.circle(preview, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(preview, label, (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    corrected, binary = illumination_correction(rectified)
    for name, im in [('corner_detection.jpg', preview), ('rectified_document.png', rectified), ('corrected_document.png', corrected), ('binary_document.png', binary)]:
        save_image(output/name, im)
    S = np.diag([210.0/(WIDTH-1), 297.0/(HEIGHT-1), 1.0])
    info = {'source_image': str(source), 'corner_method': method, 'requires_visual_review': True, 'source_size_px': [image.shape[1], image.shape[0]], 'document_size_px': [WIDTH, HEIGHT], 'document_size_mm': [210, 297], 'corner_order': ['TL', 'TR', 'BR', 'BL'], 'corners_px': src.tolist(), 'image_to_rectified': H.tolist(), 'rectified_to_image': np.linalg.inv(H).tolist(), 'image_to_document_mm': (S@H).tolist(), 'rectified_to_document_mm': S.tolist()}
    (output/'document_transform.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('SCAN COMPLETE (review corner_detection.jpg before use):', output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
