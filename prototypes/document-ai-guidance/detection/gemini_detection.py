"""Detect document fields using Gemini or visualize an existing JSON response.
Run from the project folder. No live navigation is authorized by this module.
"""
import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
CLASSES = ('check_box', 'input_field', 'signature_field')
COLORS = {'check_box': '#b51fe4', 'input_field': '#126aff', 'signature_field': '#15964d'}
PROMPT = '''Analyze this Korean administrative document. Detect all writable fields and classify them into check_box, input_field, signature_field. Include checkbox symbols such as □ and [ ]. Exclude ordinary text and decorative borders. Do not invent fields. For a checkbox, box only the markable symbol, not the option text. For input fields, box the actual writable blank area or cell, not its label. For a signature field, box the area intended for a signature or seal. Use Korean labels based on nearby printed text. Return each field once. box_2d must be [y_min, x_min, y_max, x_max] normalized to 0..1000 relative to the entire supplied image. Do not add explanations.'''
SCHEMA = {'type':'object', 'required':['boxes'], 'properties':{'boxes':{'type':'array','items':{'type':'object','required':['class_name','label','box_2d'], 'properties':{'class_name':{'type':'string','enum':list(CLASSES)}, 'label':{'type':'string'}, 'box_2d':{'type':'array','items':{'type':'integer'},'description':'Exactly four integers [ymin,xmin,ymax,xmax], each 0..1000.'}}}}}}


def read_image(path):
    with Image.open(path) as source:
        # Same upright raster is used for API and overlay, avoiding EXIF mismatch.
        image = ImageOps.exif_transpose(source).convert('RGB')
    return image


def encode_image(image, path):
    buf = io.BytesIO()
    if path.suffix.lower() in ('.jpg', '.jpeg'):
        image.save(buf, format='JPEG', quality=96, subsampling=0)
        mime = 'image/jpeg'
    else:
        image.save(buf, format='PNG')
        mime = 'image/png'
    data = buf.getvalue()
    if len(data) > 14_000_000:
        raise ValueError('Image too large for this inline request. Use a smaller lossless crop or image.')
    return base64.b64encode(data).decode('ascii'), mime


def api_detect(image, path, model):
    from google import genai
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY is not set in this PowerShell session.')
    data, mime = encode_image(image, path)
    client = genai.Client(api_key=key)
    started = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        input=[{'type':'text', 'text':PROMPT}, {'type':'image', 'data':data, 'mime_type':mime}],
        response_format={'type':'text','mime_type':'application/json','schema':SCHEMA},
    )
    seconds = round(time.perf_counter()-started, 3)
    usage = getattr(interaction, 'usage', None)
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump(mode='json', exclude_none=True)
    elif usage is not None and not isinstance(usage, dict):
        usage = str(usage)
    raw = interaction.output_text
    if not raw:
        raise RuntimeError('Empty Gemini response. Check API status, quota and model settings.')
    return raw, usage, seconds


def convert_boxes(raw, width, height):
    rows = raw.get('boxes') if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError('Expected a JSON array or an object with a boxes array.')
    detections, rejected = [], []
    for i, row in enumerate(rows, 1):
        try:
            cls, label, box = row['class_name'], row['label'], row['box_2d']
            if cls not in CLASSES or not isinstance(label, str):
                raise ValueError('Invalid class or label')
            if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in box):
                raise ValueError('Invalid normalized box')
            y1, x1, y2, x2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Reversed or empty box')
            x1, x2 = x1*width/1000, x2*width/1000
            y1, y2 = y1*height/1000, y2*height/1000
            detections.append({'id': f'F{i:03}', 'class_name':cls, 'label':label,
                'box_2d':box, 'bbox_xyxy_px':[round(v,3) for v in (x1,y1,x2,y2)],
                'center_px':{'x':round((x1+x2)/2,3),'y':round((y1+y2)/2,3)},
                'confidence':None, 'navigation_verified':False})
        except (TypeError, KeyError, ValueError) as exc:
            rejected.append({'row':i,'reason':str(exc),'data':row})
    return detections, rejected


def draw_boxes(image, detections):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    line_width = max(2, round(min(width,height)/450))
    for d in detections:
        x1,y1,x2,y2 = [round(v) for v in d['bbox_xyxy_px']]
        x1,x2 = min(width-1,x1),min(width-1,x2)
        y1,y2 = min(height-1,y1),min(height-1,y2)
        color = COLORS[d['class_name']]
        draw.rectangle((x1,y1,x2,y2), outline=color,width=line_width)
        draw.text((x1,max(0,y1-15)),d['id'],fill=color,stroke_width=1,stroke_fill='white')
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=HERE/'test_document.jpg')
    parser.add_argument('--model', default=os.environ.get('GEMINI_MODEL','gemini-4-hydrogen'))
    parser.add_argument('--from-json', type=Path, help='Existing Gemini JSON; no API call. Same input image required.')
    parser.add_argument('--output', type=Path, default=HERE/'gemini_output')
    args = parser.parse_args()
    image = read_image(args.image)
    width,height = image.size
    out = args.output / (args.image.stem + ('_offline' if args.from_json else ''))
    out.mkdir(parents=True, exist_ok=True)
    print(f'Image: {args.image} ({width} x {height})', flush=True)
    if args.from_json:
        raw = args.from_json.read_text(encoding='utf-8-sig')
        usage, latency = None, None
    else:
        print(f'Calling {args.model} ...', flush=True)
        raw,usage,latency = api_detect(image,args.image,args.model)
    (out/'gemini_raw.json').write_text(raw,encoding='utf-8')
    fields,rejected = convert_boxes(json.loads(raw),width,height)
    counts = Counter(f['class_name'] for f in fields)
    result = {'source':'imported_json' if args.from_json else 'gemini',
              'model':None if args.from_json else args.model,
              'image':str(args.image.resolve()),'image_width':width,'image_height':height,
              'coordinate_system':'EXIF-upright input image pixels, top-left origin, xyxy edges',
              'navigation_verified':False,'total':len(fields),'class_counts':{k:counts[k] for k in CLASSES},
              'detections':fields,'rejected':rejected,'usage':usage,'latency_seconds':latency}
    (out/'detection_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    draw_boxes(image,fields).save(out/'detection_visualization.jpg',quality=95)
    print('\n========== GEMINI DETECTION ==========')
    for field in fields:
        print(f"{field['id']} {field['class_name']:16} {field['label']} | {field['center_px']}")
    print(f'Total: {len(fields)} | Rejected: {len(rejected)}')
    print(f'Counts: {result["class_counts"]}\nLatency: {latency}\nUsage: {usage}\nSaved: {out}')
    print('Coordinates are candidates only; verify before finger navigation.')


if __name__ == '__main__':
    main()
