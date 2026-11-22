"""등록 양식 없이 촬영한 페이지의 OCR·체크 후보·질문을 제공한다."""
from __future__ import annotations

import re
import secrets
import threading
from collections import defaultdict

import cv2
import numpy as np

from docagent.home.capture import CaptureCoach
from docagent.home.framing import crop_document, document_boundary
from docagent.home.questions import QuestionUnavailable
from docagent.home.server import HomeApplication, decode_image


def position(box, shape):
    x, y, w, h = box
    height, width = shape[:2]
    col = min(2, int((x + w / 2) / width * 3))
    row = min(2, int((y + h / 2) / height * 3))
    return f"문서 {['위쪽', '가운데', '아래쪽'][row]} {['왼쪽', '가운데', '오른쪽'][col]}"


def join_ocr_words(words):
    """OCR이 한글 음절마다 만든 공백을 실제 글자 상자 간격으로 복원한다."""
    if not words:
        return ''
    threshold = float(np.median([w['box'][3] for w in words])) * .28
    text, right = '', None
    for word in words:
        x, _, width, _ = word['box']
        if right is not None and x - right > threshold:
            text += ' '
        text += word['text']
        right = max(right or 0, x + width)
    return text


def checkbox_candidates(gray, lines):
    """사각 테두리+주변 OCR이 있는 후보만 보고한다. 손글씨/선택 의미는 추정하지 않는다."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    short = min(gray.shape)
    found = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        if not max(10, short * .008) <= min(w, h) <= short * .04 or not .75 < w / h < 1.33:
            continue
        poly = cv2.approxPolyDP(contour, .035 * cv2.arcLength(contour, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        if any(abs(x + w / 2 - a['box'][0] - a['box'][2] / 2) < w * .65
               and abs(y + h / 2 - a['box'][1] - a['box'][3] / 2) < h * .65 for a in found):
            continue
        nearby = [line for line in lines if abs(line['box'][1] + line['box'][3] / 2 - y - h / 2) < h * 1.3
                  and line['box'][0] + line['box'][2] > x + w and line['box'][0] < x + w + gray.shape[1] * .35]
        if not nearby:
            continue
        inset = max(3, round(min(w, h) * .25))
        interior = binary[y + inset:y + h - inset, x + inset:x + w - inset]
        if not interior.size:
            continue
        ink = float(np.mean(interior > 0))
        status = 'marked' if ink >= .12 else 'empty' if ink < .025 else 'uncertain'
        line = min(nearby, key=lambda line: abs(line['box'][1] + line['box'][3] / 2 - y - h / 2))
        if h < line['box'][3] * .75:
            continue
        words = [word for word in line.get('words', []) if word['box'][0] >= x + w - 2]
        # 같은 행에 체크칸이 여러 개 있으면 다음 칸 전의 가까운 글자만 연결한다.
        labels, end = [], x + w
        for word in words:
            if word['box'][0] - end > max(w * 2, gray.shape[1] * .06): break
            if re.search(r'[가-힣A-Za-z0-9]', word['text']): labels.append(word)
            end = word['box'][0] + word['box'][2]
        if not labels:
            continue
        label = join_ocr_words(labels)
        found.append({'box': [x, y, w, h], 'label': label,
                      'status': status, 'ink_ratio': round(ink, 3), 'position': position((x, y, w, h), gray.shape)})
    return sorted(found, key=lambda a: (a['box'][1], a['box'][0]))[:40]


def recognize_lines(gray):
    """가로 쓰기 문서를 행 단위로 읽어 한국어를 세로 글자로 오인하는 현상을 줄인다."""
    import pytesseract
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)
    height, width = gray.shape
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(80, width // 10)), np.uint8))
    rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(100, height // 10), 1), np.uint8))
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    clean = gray.copy(); clean[rules > 0] = 255; ink[rules > 0] = 0
    rows = np.flatnonzero(np.count_nonzero(ink, axis=1) > max(2, width * .0015))
    bands = []
    for row in rows:
        if bands and row - bands[-1][1] <= 4:
            bands[-1][1] = int(row)
        else:
            bands.append([int(row), int(row)])
    if not bands or len(bands) > 80 or any(b - a > height * .08 for a, b in bands):
        return pytesseract.image_to_data(clean, lang='kor', config='--psm 6',
                                        output_type=pytesseract.Output.DICT, timeout=25)
    fields = ('text', 'conf', 'left', 'top', 'width', 'height', 'block_num', 'par_num', 'line_num')
    result = {key: [] for key in fields}
    for number, (top, bottom) in enumerate(bands):
        crop = cv2.copyMakeBorder(clean[top:bottom + 1], 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        data = pytesseract.image_to_data(crop, lang='kor', config='--psm 7',
                                        output_type=pytesseract.Output.DICT, timeout=8)
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
            for key in fields:
                value = data[key][i]
                if key == 'top': value += top - 12
                elif key == 'left': value -= 12
                elif key == 'block_num': value = number
                elif key in ('par_num', 'line_num'): value = 1
                result[key].append(value)
    return result


def read_page(image, mode='photo', ocr=None):
    boundary = ocr.boundary(image) if ocr and mode == 'photo' else document_boundary(image)
    page = crop_document(image, boundary) if mode == 'photo' else image
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(2., 2400 / max(h, w))
    if scale != 1:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    # 확대했더라도 원본에 없던 글자 세부 정보가 생기는 것은 아니다.
    warnings = []
    if min(h, w) < 650:
        warnings.append('글자가 작게 찍혔습니다. 읽힌 내용이 이상하면 문서를 조금 크게 촬영해 주세요.')
    if mode == 'photo' and boundary['status'] != 'framed':
        warnings.append('문서 전체가 담겼는지는 확인하지 못했습니다. 사진에 보이는 범위만 읽었습니다.')
    if ocr:
        # 원근 변환은 작은 인쇄 글자를 보간하며 손상시킬 수 있다. Vision에는
        # 촬영 원본을 주고 문서 밖의 결과를 좌표로 제외한다.
        native_page = image
        ratio = min(1., 4000 / max(image.shape[:2]))
        if ratio < 1:
            native_page = cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)))
        lines = ocr.read(native_page)
        gray = cv2.cvtColor(native_page, cv2.COLOR_BGR2GRAY)
        if mode == 'photo' and boundary['status'] == 'framed':
            q = np.asarray(boundary['quad'], np.float32) * ratio
            lines = [line for line in lines if cv2.pointPolygonTest(q,
                (line['box'][0] + line['box'][2] / 2, line['box'][1] + line['box'][3] / 2), False) >= 0]
            x, y, width, height = cv2.boundingRect(q)
            x, y = max(0, x), max(0, y)
            gray = gray[y:min(gray.shape[0], y + height), x:min(gray.shape[1], x + width)]
            for line in lines:
                line['box'][0] -= x; line['box'][1] -= y
                for word in line.get('words', []):
                    word['box'][0] -= x; word['box'][1] -= y
        # 열별 반환 순서가 표의 왼쪽 설명을 전부 먼저 읽지 않도록 위치 순서를 복원한다.
        lines.sort(key=lambda line: (line['box'][1] + line['box'][3] / 2, line['box'][0]))
        lines = [{**line, 'id':f'S{i + 1}', 'page':1, 'origin':'ocr'} for i, line in enumerate(lines)]
    else:
        data = recognize_lines(gray)
        buckets = defaultdict(list)
        for i, text in enumerate(data['text']):
            text = text.strip()
            if text and float(data['conf'][i]) >= 0:
                buckets[(data['block_num'][i], data['par_num'][i], data['line_num'][i])].append(i)
        lines = []
        for indices in buckets.values():
            words = [{'text': data['text'][i].strip(), 'box': [data['left'][i], data['top'][i], data['width'][i], data['height'][i]]} for i in indices]
            text = join_ocr_words(words)
            if not re.search(r'[가-힣A-Za-z0-9]', text):
                continue
            x, y = min(data['left'][i] for i in indices), min(data['top'][i] for i in indices)
            right = max(data['left'][i] + data['width'][i] for i in indices)
            bottom = max(data['top'][i] + data['height'][i] for i in indices)
            confidence = sum(max(0., float(data['conf'][i])) * len(data['text'][i]) for i in indices) / sum(len(data['text'][i]) for i in indices)
            lines.append({'id': f'S{len(lines) + 1}', 'text': text, 'page': 1,
                          'confidence': round(confidence / 100, 2),
                          'uncertain': any(float(data['conf'][i]) < 60 for i in indices),
                          'box': [x, y, right - x, bottom - y], 'origin': 'ocr',
                          'words': words})
    if not lines or sum(len(line['text']) for line in lines) < 4:
        return {'status': 'unreadable', 'text': '', 'lines': [], 'checkboxes': [], 'signatures': [],
                'message': '읽을 수 있는 글자를 찾지 못했어요. 초점을 맞추거나 가까이에서 다시 촬영해 주세요.'}
    readable, evidence = text_readability(lines)
    if not readable:
        return {'status': 'unreadable', 'text': '', 'lines': [], 'checkboxes': [], 'signatures': [],
                'engine': getattr(ocr, 'name', 'tesseract'), 'quality': evidence,
                'message': '글자를 대부분 정확히 읽지 못했어요. 문서를 크게 보이게 하거나 조명과 초점을 조절해 다시 촬영해 주세요.'}
    uncertain = sum(line['uncertain'] for line in lines)
    if uncertain:
        warnings.append(f'{len(lines)}개 줄 중 {uncertain}개 줄에 인식이 불확실한 글자가 있습니다.')
    for line in lines:
        if re.search(r'서\s*명|날\s*인', line['text']):
            line['position'] = position(line['box'], gray.shape)
    signatures = [{'text': line['text'], 'position': position(line['box'], gray.shape)}
                  for line in lines if re.search(r'[(（]\s*서\s*명\s*[)）]|신\s*청\s*인.*서\s*명|^서\s*명[ :：_]*$', line['text'])]
    boxes = checkbox_candidates(gray, lines)
    return {'status': 'read', 'text': '\n'.join(line['text'] for line in lines), 'lines': lines,
            'checkboxes': boxes, 'signatures': signatures, 'warnings': warnings,
            'message': '사진에서 인식한 글자입니다. 인식 오류나 누락이 있을 수 있습니다.',
            'boundary_confirmed': mode == 'scan' or boundary['status'] == 'framed',
            'engine': getattr(ocr, 'name', 'tesseract'), 'quality': evidence,
            'page_size_px': [gray.shape[1], gray.shape[0]]}


def text_readability(lines):
    total = sum(len(line['text']) for line in lines)
    trusted = sum(len(line['text']) for line in lines if not line['uncertain'] and line['confidence'] >= .6)
    ratio = trusted / max(1, total)
    average = sum(line['confidence'] * len(line['text']) for line in lines) / max(1, total)
    evidence = {'reliable_text_ratio':round(ratio,3), 'mean_confidence':round(average,3),
                'uncertain_lines':sum(line['uncertain'] for line in lines), 'line_count':len(lines)}
    return average >= .5 and ratio >= .45, evidence


def question_context(question, references, history, budget=3000):
    """긴 문서는 현재 질문·직전 질문에 관련된 줄과 이웃 줄을 로컬에서 검색한다."""
    compact = [{key: line[key] for key in ('id', 'text', 'page', 'uncertain', 'origin', 'position') if key in line} for line in references]
    import json
    cost = lambda item: len(json.dumps(item, ensure_ascii=False).encode())
    if sum(map(cost, compact)) < budget:
        return compact, False
    query = question + ' ' + (history[-1]['question'] if history else '')
    query = re.sub(r'\s+', '', query.lower())
    grams = set(query[i:i + 2] for i in range(len(query) - 1))
    previous = set(history[-1].get('source_ids', [])) if history else set()
    scored = sorted(range(len(compact)), key=lambda i: (
        sum(g in compact[i]['text'].lower().replace(' ', '') for g in grams) + 3 * (compact[i]['id'] in previous)
    ), reverse=True)
    picked, used = set(), 0
    for index in scored:
        for i in (index, index - 1, index + 1):
            if not 0 <= i < len(compact) or i in picked:
                continue
            size = cost(compact[i])
            if used + size <= budget:
                picked.add(i); used += size
    return [compact[i] for i in sorted(picked)], True


class ReaderApplication(HomeApplication):
    """기본 노트북 모드. 기존 승인 양식·모바일 API와 독립된 OCR 세션."""
    def __init__(self, *, questions=None, speech=None, ocr=None):
        self.template = None
        self.questions, self.speech = questions, speech
        self.ocr = ocr
        self.coach = CaptureCoach(boundary_detector=ocr.boundary if ocr else None)
        self.token, self.lock = secrets.token_urlsafe(32), threading.Lock()
        self.rate = 1.
        self.index = 0
        self.reset()

    def reset(self):
        self.coach.reset()
        self.capture = self.document = self.capture_ocr = self.capture_id = None
        self.observation = self.content_check = None
        self.document_ready = False
        self.question_history = []
        self.index = 0

    def state(self):
        return {'mode': 'ocr', 'purpose': 'general', 'title': '생활 문서 도우미',
                'session': {'preferences': {'speech_rate': self.rate}, 'groups': [],
                            'current_group': None, 'current_index': self.index, 'complete': False},
                'document_ready': self.document_ready, 'has_capture': self.capture is not None,
                'can_verify': self.document_ready, 'document': self.document, 'capture_id': self.capture_id,
                'capture_ocr': self.capture_ocr, 'content_check': None,
                'questions_enabled': self.questions is not None, 'speech_enabled': self.speech is not None,
                'question_model': self.questions.model if self.questions else None,
                'external_transfer_enabled': False, 'session_saved': False, 'ocr_engine': getattr(self.ocr, 'name', 'tesseract')}

    def handle(self, action, data):
        if action == 'reset_capture':
            self.reset()
            return self.state()
        if action == 'capture':
            self.reset()
            image = decode_image(data.get('image'))
            mode = data.get('mode', 'photo')
            if mode not in ('photo', 'scan'):
                raise ValueError('촬영 형식을 확인하세요.')
            self.capture = image
            try:
                self.capture_ocr = read_page(image, mode, self.ocr)
            except (RuntimeError, OSError) as exc:
                raise QuestionUnavailable('문자 인식을 완료하지 못했어요. 다시 촬영하거나 Tesseract 한국어 설치를 확인해 주세요.') from exc
            if self.capture_ocr['status'] != 'read':
                return {**self.state(), 'instruction': self.capture_ocr['message']}
            self.capture_id = secrets.token_hex(12)
            self.document_ready = True
            ocr = self.capture_ocr
            summary = f"사진에서 {len(ocr['lines'])}개 줄의 글자를 읽었습니다."
            if ocr['signatures']:
                summary += f" 서명란으로 보이는 문구 {len(ocr['signatures'])}개가 있습니다. " + ' '.join(f"{s['position']}에 {s['text']}." for s in ocr['signatures'][:2])
            if ocr['checkboxes']:
                summary += f" 체크칸 후보 {len(ocr['checkboxes'])}개를 찾았습니다."
            summary += ' ' + ' '.join(ocr['warnings'])
            self.document = {'summary': summary.strip(), 'references': ocr['lines'], 'ocr': ocr,
                             'capture_id': self.capture_id}
            return {**self.state(), 'instruction': summary + ' 인식한 원문 듣기, 질문하기, 체크 상태 듣기를 사용할 수 있습니다.'}
        if action == 'preferences':
            rate = data.get('speech_rate')
            if type(rate) not in (int, float) or rate not in (.8, 1., 1.5, 2.):
                raise ValueError('읽기 속도를 확인하세요.')
            self.rate = rate
            return self.state()
        if action in ('read', 'move', 'verify'):
            if not self.document_ready:
                raise ValueError('먼저 촬영해 주세요.')
            if action == 'verify':
                boxes = self.capture_ocr['checkboxes']
                message = '체크칸 후보를 찾지 못했습니다. 체크칸이 없다는 뜻은 아닙니다.' if not boxes else (
                    '사진에서 찾은 체크칸 후보입니다. 인쇄된 기호나 글자를 체크로 잘못 볼 수 있습니다.\n' +
                    '\n'.join(f"{b['position']}, {b['label']}: " +
                              {'marked': '표시 있음으로 추정', 'empty': '비어 있음으로 추정', 'uncertain': '표시 판독 불가'}[b['status']]
                              for b in boxes))
                return {**self.state(), 'instruction': message}
            lines = self.capture_ocr['lines']
            if action == 'move':
                self.index = max(0, min(len(lines) - 1, self.index + {'previous': -1, 'next': 1}.get(data.get('direction'), 0)))
            selected = lines if data.get('scope') == 'page' else [lines[self.index]]
            text = '\n'.join(('인식 불확실. ' if line['uncertain'] else '') + line['text'] for line in selected)
            return {**self.state(), 'instruction': '사진에서 인식한 내용입니다.\n' + text}
        if action == 'ask':
            if not self.document_ready or not self.questions or data.get('capture_id') != self.capture_id:
                raise ValueError('현재 문서를 촬영한 뒤 질문해 주세요.')
            question = data.get('question')
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
                raise ValueError('질문은 1~500자로 입력해 주세요.')
            references, limited = question_context(question.strip(), self.document['references'], self.question_history)
            history = [{'question': h['question'][:100], 'source_ids': h['source_ids']} for h in self.question_history[-2:]]
            result = self.questions.answer(question.strip(), references, history)
            if result['status'] == 'found':
                result['answer'] = '사진에서 인식한 내용 중 관련 문구입니다.\n' + '\n\n'.join(
                    ('인식이 불확실한 글자가 포함되어 있습니다. ' if s.get('uncertain') else '') + s['text'] + (f". 문구 위치: {s['position']}" if s.get('position') else '') for s in result['sources'])
            elif result['status'] == 'not_found':
                result['answer'] = '인식한 글자' + ('에서 검색한 부분' if limited else '') + '에서 답을 찾지 못했습니다. 원문 듣기로 확인하거나 질문을 구체적으로 바꿔 주세요.'
            result['retrieval_limited'] = limited
            self.question_history = (self.question_history + [{'question': question.strip(), 'source_ids': [s['id'] for s in result['sources']]}])[-3:]
            return {**self.state(), 'answer': result, 'instruction': result['answer']}
        if action not in ('frame', 'guide', 'privacy', 'transcribe'):
            raise ValueError('지원하지 않는 명령입니다.')
        return super().handle(action, data)
