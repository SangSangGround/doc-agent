"""PaddleOCR 3.3 test. Click document corners TL, TR, BR, BL, then Space."""
import json
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
IMAGE = BASE.parent / 'detection' / 'test_document.jpg'
W, H = 2100, 2970  # A4, 10 px/mm


def save_image(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    data.tofile(str(path))


def select_corners(image):
    h, w = image.shape[:2]
    scale = min(1000 / w, 750 / h, 1.0)
    preview = cv2.resize(image, (round(w * scale), round(h * scale)))
    ph, pw = preview.shape[:2]
    points = []
    name = 'Select TL TR BR BL | SPACE confirm | R reset | Q quit'
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.setMouseCallback(name, click)
    try:
        while True:
            display = preview.copy()
            for i, p in enumerate(points):
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                cv2.putText(display, str(i+1), (p[0]+7, p[1]+15),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            if len(points) > 1:
                cv2.polylines(display, [np.array(points, np.int32)],
                              len(points) == 4, (0, 255, 0), 2)
            cv2.imshow(name, display)
            key = cv2.waitKey(30) & 255
            if key == ord('q') or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit('Cancelled')
            if key == ord('r'):
                points.clear()
            if key == 32 and len(points) == 4:
                p = np.array(points, np.float32)
                if cv2.isContourConvex(p) and cv2.contourArea(p, oriented=True) > pw*ph*.05:
                    return p * np.array([w/pw, h/ph], np.float32)
                print('Invalid corners. Press R and select again.')
    finally:
        cv2.destroyAllWindows()


def main():
    if not IMAGE.is_file():
        raise FileNotFoundError(IMAGE)
    image = cv2.imdecode(np.fromfile(str(IMAGE), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Cannot read document')
    corners = select_corners(image)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    page = cv2.warpPerspective(image, matrix, (W,H))
    save_image(BASE/'rectified_document.png', page)
    print('Loading PaddleOCR...')
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='korean', ocr_version='PP-OCRv5', device='cpu',
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, enable_mkldnn=False,
                    text_det_limit_side_len=2970, text_det_limit_type='max',
                    text_rec_score_thresh=0.0)
    results = list(ocr.predict(page))
    if len(results) != 1:
        raise RuntimeError(f'Expected one page, got {len(results)}')
    result = results[0]
    texts, scores, polys = result['rec_texts'], result['rec_scores'], result['rec_polys']
    if not len(texts) == len(scores) == len(polys):
        raise RuntimeError('OCR output lengths do not match')
    display = page.copy()
    items = []
    for text, score, poly in zip(texts, scores, polys):
        if not str(text).strip():
            continue
        p = np.asarray(poly, dtype=np.float32).reshape(-1,2)
        p[:,0] = np.clip(p[:,0],0,W-1)
        p[:,1] = np.clip(p[:,1],0,H-1)
        x1,y1 = np.floor(p.min(axis=0)).astype(int).tolist()
        x2,y2 = np.ceil(p.max(axis=0)).astype(int).tolist()
        confidence = float(score)*100
        item = {'id':len(items)+1,'text':str(text),'confidence':confidence,
                'bbox':[x1,y1,x2,y2],'center':[(x1+x2)/2,(y1+y2)/2],
                'polygon':p.tolist(),
                'bbox_mm':[x1*210/(W-1),y1*297/(H-1),x2*210/(W-1),y2*297/(H-1)],
                'needs_review':confidence<80}
        items.append(item)
        color = (0,160,0) if confidence>=80 else (0,140,255)
        cv2.polylines(display,[np.rint(p).astype(np.int32)],True,color,3)
        cv2.putText(display,str(item['id']),(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
        print(f"[{item['id']}] {text} / confidence={confidence:.1f}")
    output = {'image_width':W,'image_height':H,'coordinate_unit':'pixel',
              'coordinate_frame':'rectified_document.png','paper_size_mm':[210,297],
              'original_corners_px':corners.tolist(),'original_to_rectified':matrix.tolist(),
              'total_texts':len(items),'texts':items}
    (BASE/'paddle_ocr_result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    (BASE/'paddle_ocr_text.txt').write_text('\n'.join(i['text'] for i in items),encoding='utf-8')
    save_image(BASE/'paddle_ocr_visualization.jpg',display)
    print(f'Complete: {len(items)} text regions. Outputs: {BASE}')

if __name__ == '__main__':
    main()
