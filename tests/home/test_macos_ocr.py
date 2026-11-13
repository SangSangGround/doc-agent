import copy

import cv2
import numpy as np
import pytest

from docagent.home.macos_ocr import MacVision, NativeOcrUnavailable, boundary_result
from docagent.home.reader import read_page, text_readability


def test_document_segmentation_keeps_all_four_corners_and_rejects_clipping():
    observation={'confidence':.99,'quad':[[50,180],[420,190],[440,720],[65,745]]}
    result=boundary_result([observation],(1920,1080),800/1920)
    assert result['status']=='framed'
    assert result['quad'][0]==[120.,432.]
    observation['quad'][1][0]=450
    assert boundary_result([observation],(1920,1080),800/1920)['status']=='clipped'
    assert boundary_result([], (1920,1080),1)['status']=='searching'
    observation['confidence']=.3
    assert boundary_result([observation],(1920,1080),1)['status']=='searching'


def test_low_confidence_page_is_not_success_because_one_title_was_read():
    lines=[{'text':'선명한 제목','confidence':.95,'uncertain':False}]
    lines += [{'text':'읽지 못한 본문 내용 ' * 5,'confidence':.3,'uncertain':True} for _ in range(12)]
    ready, quality = text_readability(lines)
    assert not ready and quality['reliable_text_ratio'] < .1
    assert text_readability([{'text':'모임은 토요일입니다.','confidence':.98,'uncertain':False}])[0]


def test_native_ocr_uses_original_pixels_filters_background_and_keeps_layout():
    image=np.full((900,900,3),245,np.uint8)
    image[200:205,200:205]=10
    class FakeVision:
        name='apple_vision'
        def boundary(self, frame):
            return {'status':'framed','quad':[[100,100],[800,110],[780,800],[100,800]],'area_ratio':.6}
        def read(self, frame):
            assert np.array_equal(frame,image), '원근 보정본 대신 원본을 전달해야 한다'
            return [
                {'text':'두 번째 문장','box':[300.,400.,200.,30.],'confidence':.99,'uncertain':False,'words':[]},
                {'text':'배경의 글자','box':[10.,10.,60.,20.],'confidence':.99,'uncertain':False,'words':[]},
                {'text':'첫 번째 문장','box':[200.,200.,200.,30.],'confidence':.99,'uncertain':False,'words':[]}]
    result=read_page(image,'photo',FakeVision())
    assert result['status']=='read' and result['engine']=='apple_vision'
    assert result['text']=='첫 번째 문장\n두 번째 문장'
    assert result['lines'][0]['box'][:2]==[100.,100.]


def test_bad_native_coordinates_fail_closed(monkeypatch):
    reader=MacVision(binary='unused')
    monkeypatch.setattr(reader,'_run',lambda *_: {'engine':'apple_vision','lines':[
        {'text':'글자','confidence':1,'box':[float('nan'),0,10,10],'words':[]}]})
    with pytest.raises(NativeOcrUnavailable):reader.read(np.zeros((10,10,3),np.uint8))
