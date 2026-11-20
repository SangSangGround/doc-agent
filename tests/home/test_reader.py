"""일반 문서 촬영·OCR 세션의 경계와 상태 회귀 검사."""
import copy
import os

import cv2
import numpy as np
import pytest

from docagent.home.capture import CaptureCoach
from docagent.home.framing import document_boundary
from docagent.home.reader import ReaderApplication, question_context, read_page
from docagent.home.server import encode_image


def paper(w=500, h=650, background=100, rotate=0):
    image = np.full((850, 950, 3), background, np.uint8)
    x, y = (950 - w) // 2, (850 - h) // 2
    image[y:y + h, x:x + w] = 245
    for row in range(y + 50, y + h - 30, 50):
        cv2.putText(image, 'HOME NOTICE 123', (x + 20, row), cv2.FONT_HERSHEY_SIMPLEX, .6, (10, 10, 10), 1)
    if rotate:
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((475, 425), rotate, 1.),
                               (950, 850), borderValue=(background,) * 3)
    return image


@pytest.mark.parametrize('kwargs', [{}, {'w':650,'h':400}, {'w':420,'h':420},
                                    {'background':225}, {'rotate':16}])
def test_general_page_reports_framing_and_sharpness(kwargs):
    image = paper(**kwargs)
    coach = CaptureCoach()
    for stamp in (0., .35, .75):
        result = coach.update(image, timestamp=stamp)
    assert result['ready_to_capture'], result
    assert result['boundary']['status'] == 'framed'
    assert result['focus_clear']


def test_blurred_text_is_not_ready_for_focus_even_when_paper_is_in_frame():
    result = CaptureCoach().update(cv2.GaussianBlur(paper(), (15, 15), 4))
    assert result['boundary']['status'] == 'framed'
    assert not result['focus_clear']


def test_exposure_change_does_not_reset_corner_stability():
    image = paper()
    coach = CaptureCoach()
    coach.update(image, timestamp=0.)
    coach.update(cv2.convertScaleAbs(image, alpha=.90), timestamp=.35)
    assert coach.update(cv2.convertScaleAbs(image, alpha=.98), timestamp=.75)['ready_to_capture']


def test_cutoff_or_uniform_surface_is_not_framed():
    for image in (paper()[:, 290:], np.full((800, 1000, 3), 245, np.uint8)):
        coach = CaptureCoach()
        for stamp in (0., .4, .9):
            assert not coach.update(image, timestamp=stamp)['ready_to_capture']


def sample_ocr():
    return {'status':'read','text':'준비물은 물입니다.', 'lines':[
        {'id':'S1','text':'준비물은 물입니다.','page':1,'uncertain':False,'origin':'ocr','confidence':.95}],
        'checkboxes':[], 'signatures':[], 'warnings':[]}


def captured(monkeypatch):
    monkeypatch.setattr('docagent.home.reader.read_page', lambda *_: copy.deepcopy(sample_ocr()))
    app = ReaderApplication()
    result = app.handle('capture', {'image':encode_image(paper())})
    assert result['document_ready']
    return app


def test_arbitrary_ocr_is_source_without_registered_form(monkeypatch):
    app = captured(monkeypatch)
    assert app.template is None and app.state()['mode'] == 'ocr'
    assert '준비물은 물입니다.' in app.handle('read',{'scope':'page'})['instruction']
    assert '체크칸이 없다는 뜻은 아닙니다' in app.handle('verify',{})['instruction']
    with pytest.raises(ValueError):
        app.handle('confirm', {'group_id':'invented'})


def test_failed_recapture_discards_previous_text_and_question_id(monkeypatch):
    app = captured(monkeypatch)
    old_id = app.capture_id
    with pytest.raises(ValueError):
        app.handle('capture', {'image':'invalid'})
    assert not app.document_ready and app.capture_id is None and app.document is None
    with pytest.raises(ValueError):
        app.handle('ask', {'capture_id':old_id,'question':'준비물?'})
    with pytest.raises(ValueError):
        app.handle('read',{'scope':'page'})


def test_uncertain_ocr_is_announced_and_questions_use_only_this_capture(monkeypatch):
    app = captured(monkeypatch)
    app.capture_ocr['lines'][0]['uncertain'] = True
    assert '인식 불확실' in app.handle('read',{'scope':'page'})['instruction']
    class FakeQuestions:
        model='test'
        def answer(self, question, sources, history):
            assert sources[0]['text'] == '준비물은 물입니다.'
            return {'status':'found','sources':[sources[0]],'answer':'모델이 생성한 문장을 그대로 읽으면 안 됨'}
    app.questions = FakeQuestions()
    result = app.handle('ask',{'question':'준비물은?','capture_id':app.capture_id})
    assert '준비물은 물입니다.' in result['instruction']
    assert '인식이 불확실' in result['instruction']
    assert '모델이 생성한' not in result['instruction']


def test_long_document_retrieval_includes_relevant_line_and_is_bounded():
    lines = [{'id':f'S{i}','text':f'{i}번 공지: 아파트 일상 안내입니다.','page':1,'uncertain':False,'origin':'ocr'} for i in range(150)]
    lines[93]['text'] = '소독 일정은 11월 20일 오전입니다.'
    sources, limited = question_context('소독 일정이 언제인가요?', lines, [])
    assert limited and any(s['id'] == 'S93' for s in sources)
    import json
    assert sum(len(json.dumps(s,ensure_ascii=False).encode()) for s in sources) <= 3000


@pytest.mark.skipif(os.environ.get('DOCAGENT_TEST_OFFLINE') == '1', reason='로컬 Tesseract 실측')
def test_real_ocr_home_notice_and_checkbox_states():
    pytest.importorskip('pytesseract')
    import pytesseract
    try:
        if 'kor' not in pytesseract.get_languages(): pytest.skip('한국어 OCR 미설치')
    except pytesseract.TesseractNotFoundError:
        pytest.skip('OCR 미설치')
    from PIL import Image, ImageDraw
    from docagent.testing.fonts import load_font
    page = Image.new('RGB',(1000,1400),'white'); draw = ImageDraw.Draw(page)
    font=load_font(32)
    for y,text in [(100,'가정통신문'),(200,'준비물은 물과 수건입니다.'),(300,'모임은 토요일 오전 열 시입니다.'),(650,'신청인 (서명)')]:
        draw.text((90,y),text,font=font,fill='black')
    for y,label,marked in [(430,'참석',True),(520,'불참',False)]:
        draw.rectangle((90,y,124,y+34),outline='black',width=2)
        draw.text((150,y),label,font=font,fill='black')
        if marked: draw.line((95,y+12,103,y+25,120,y+5), fill='black',width=4)
    result=read_page(cv2.cvtColor(np.array(page),cv2.COLOR_RGB2BGR),'scan')
    assert result['status']=='read'
    assert '준비물은물과수건입니다' in ''.join(result['text'].split())
    assert len(result['checkboxes']) == 2
    assert [b['status'] for b in result['checkboxes']] == ['marked','empty']
    assert len(result['signatures']) == 1


def test_korean_syllable_spaces_follow_visible_geometry():
    from docagent.home.reader import join_ocr_words
    words = [
        {'text':'준','box':[10,0,70,30]}, {'text':'비','box':[30,0,20,30]},
        {'text':'물','box':[52,0,25,30]}, {'text':'물','box':[96,0,25,30]},
        {'text':'12,000','box':[140,0,90,30]}, {'text':'원','box':[232,0,25,30]}]
    assert join_ocr_words(words) == '준비물 물 12,000원'
