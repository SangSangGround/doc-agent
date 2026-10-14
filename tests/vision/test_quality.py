import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from docagent import analyze, check_capture
from docagent.vision.quality import assess_capture, QualityPolicy
from docagent.vision.normalize import QuadDetection


def document():
    image=np.full((1000,800,3),255,np.uint8)
    for y in range(100,900,35):
        cv2.putText(image,'SYNTHETIC FORM 123 ABC',(80,y),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
    return image


def test_sharp_scan_passes_but_photo_boundary_is_unknown():
    image=document()
    assert assess_capture(image,mode='scan').can_continue
    assert not assess_capture(image,mode='photo').can_continue


def test_blurred_and_low_resolution_inputs_request_retake():
    assert 'blurred' in assess_capture(cv2.GaussianBlur(document(),(31,31),8),mode='scan').reasons
    assert 'low_resolution' in assess_capture(cv2.resize(document(),(160,200)),mode='scan').reasons
    assert 'low_contrast' in assess_capture(np.full((1000,800),255,np.uint8),mode='scan').reasons
    assert 'too_dark' in assess_capture(np.zeros((1000,800),np.uint8),mode='scan').reasons


@pytest.mark.parametrize('image',[None,np.array([]),np.zeros((1,1),np.uint8),np.zeros((100,100),float)])
def test_invalid_image_is_not_passed(image):
    result=assess_capture(image)
    assert result.status=='retake'
    assert result.reasons==('invalid_image',)


def quad(points, confidence=.99, fallback=False):
    return QuadDetection(np.array(points,dtype=float),confidence,'test',fallback,.8,1.4)


def test_boundary_cutoff_tilt_and_uncertainty(monkeypatch):
    import docagent.vision.quality as quality
    image=document()
    safe=quad([[50,40],[750,40],[750,960],[50,960]])
    monkeypatch.setattr(quality,'detect_document_quad_detailed',lambda im:safe)
    assert assess_capture(image).status=='pass'
    monkeypatch.setattr(quality,'detect_document_quad_detailed',lambda im:replace(safe,confidence=.5))
    assert assess_capture(image).status=='review'
    cut=quad([[0,40],[750,40],[750,960],[0,960]])
    monkeypatch.setattr(quality,'detect_document_quad_detailed',lambda im:cut)
    assert 'edge_cutoff' in assess_capture(image).reasons
    tilted=quad([[50,40],[750,240],[750,960],[50,760]])
    monkeypatch.setattr(quality,'detect_document_quad_detailed',lambda im:tilted)
    assert 'tilted' in assess_capture(image).reasons


def test_quality_rejection_precedes_any_api_or_ocr_initialization(tmp_path,monkeypatch,capsys):
    path=tmp_path/'blur.png'
    path.write_bytes(cv2.imencode('.png',np.zeros((100,100,3),np.uint8))[1].tobytes())
    def forbidden(*a,**kw): pytest.fail('no setup or API call before quality gate')
    monkeypatch.setattr(analyze,'_load_dotenv',forbidden)
    monkeypatch.setattr(analyze,'build_detector',forbidden)
    assert analyze.main([str(path),'--capture-mode','photo','--detector','roboflow'])==1
    assert '다시 촬영' in capsys.readouterr().err


def test_cli_does_not_claim_uncertain_photo_passed(tmp_path,capsys):
    path=tmp_path/'image.png'
    path.write_bytes(cv2.imencode('.png',document())[1].tobytes())
    assert check_capture.main([str(path),'--mode','scan'])==0
    assert json.loads(capsys.readouterr().out)['status']=='pass'
    assert check_capture.main([str(path)])==2
    assert json.loads(capsys.readouterr().out)['can_continue'] is False


def test_policy_rejects_nan_and_bad_mode():
    with pytest.raises(ValueError): QualityPolicy(min_sharpness=float('nan'))
    with pytest.raises(ValueError): assess_capture(document(),mode='guess')
