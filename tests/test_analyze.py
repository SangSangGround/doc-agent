"""Roboflow → Crop OCR → 구조화 경로를 외부 통신 없이 관통한다."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from docagent import analyze
from docagent.config import DocAgentConfig
from docagent.contracts import FieldRole, FieldType
from docagent.detector import roboflow_detector as rf
from docagent.errors import DocumentNotFoundError, VisionError
from docagent.pipeline import build_session
from docagent.pii.image_mask import BlankDocumentDetector
from docagent.vision.detector_factory import build_detector, describe_detector
from docagent.vision.heuristic import HeuristicDetector
from docagent.vision.ocr import TesseractOcr


class WorkflowClient:
    def __init__(self, empty=False):
        self.calls = []
        self.empty = empty

    def run_workflow(self, **kwargs):
        self.calls.append(kwargs)
        image = kwargs["images"]["image"]
        assert image.shape == (2970, 2100, 3)  # pipeline normalized to 254 dpi
        predictions = [] if self.empty else [
            dict(x=625, y=1025, width=50, height=50, confidence=0.95, **{"class": "check_box"}),
            dict(x=1125, y=1025, width=50, height=50, confidence=0.96, **{"class": "check_box"}),
            dict(x=1150, y=1850, width=300, height=100, confidence=0.94, **{"class": "signature_field"}),
        ]
        return [{"predictions": {"predictions": predictions}}]


@pytest.fixture
def fake_tesseract(monkeypatch):
    calls = []
    def image_to_data(image, **kwargs):
        calls.append(image.shape)
        assert kwargs['lang'] == 'kor+eng'
        if len(calls) % 2 == 1:
            return dict(text=["동의함", "동의하지 않음"], conf=[97, 96],
                        left=[520, 1020], top=[150, 150], width=[100, 250], height=[40, 40])
        return dict(text=["신청인"], conf=[93], left=[300], top=[150], width=[100], height=[40])
    sdk = SimpleNamespace(image_to_data=image_to_data, Output=SimpleNamespace(DICT="dict"),
                          get_languages=lambda **kw: ["kor", "eng"])
    monkeypatch.setattr(TesseractOcr, '_import_pytesseract', staticmethod(lambda: sdk))
    return calls


@pytest.fixture
def blank_image():
    return np.full((594, 420, 3), 255, dtype=np.uint8)


def test_pipeline_real_adapter_and_crop_ocr(blank_image, fake_tesseract):
    client = WorkflowClient()
    session = build_session(
        blank_image, detector=BlankDocumentDetector(rf.RoboflowDetector(client=client)),
        config=DocAgentConfig(dpi=254, ocr_kind="tesseract", ocr_mode="regions"),
    )
    assert len(client.calls) == 1
    assert len(fake_tesseract) == 2
    choice, signature = session.structure.fields
    assert [o.label for o in choice.options] == ["동의함", "동의하지 않음"]
    assert signature.type is FieldType.SIGNATURE
    assert signature.role is FieldRole.APPLICANT
    assert signature.box_mm.x_mm == pytest.approx(100)
    report = analyze.build_report(session)
    assert report["detections"][2]["confidence"] == 0.94
    assert report["ocr_words"][2]["confidence"] == 0.93
    assert report["coordinate_system"] == "normalized_a4_mm"
    assert report["needs_review"]  # Crop cannot certify whole-document coverage
    assert session.audit.counters()["pii_leaked"] == 0


def test_no_detections_in_regions_reports_error(blank_image, fake_tesseract):
    with pytest.raises(DocumentNotFoundError):
        build_session(blank_image, detector=BlankDocumentDetector(rf.RoboflowDetector(client=WorkflowClient(empty=True))),
                      config=DocAgentConfig(dpi=254, ocr_kind="tesseract", ocr_mode="regions"))
    assert fake_tesseract == []


def test_regions_rejects_incompatible_ocr(blank_image):
    with pytest.raises(VisionError, match="read_regions"):
        build_session(blank_image, detector=BlankDocumentDetector(rf.RoboflowDetector(client=WorkflowClient())),
                      config=DocAgentConfig(dpi=254, ocr_mode="regions"))


def test_factory_roboflow_is_explicit_and_uses_adapter_defaults(monkeypatch):
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return "fake detector"
    monkeypatch.setattr(rf, "RoboflowDetector", constructor)
    assert isinstance(build_detector(), HeuristicDetector)
    assert calls == []
    assert build_detector(prefer="roboflow") == "fake detector"
    assert calls == [{"conf": 0.5}]
    build_detector(prefer="roboflow", conf=0.3)
    assert calls[-1]["conf"] == 0.3


def test_factory_does_not_fallback_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        build_detector(prefer="roboflow")
    assert "Roboflow" in describe_detector(rf.RoboflowDetector(client=WorkflowClient()))


def test_config_ocr_roundtrip_and_environment():
    cfg = DocAgentConfig.from_env({"DOCAGENT_DETECTOR_PREFER": "roboflow",
                                  "DOCAGENT_OCR_KIND": "tesseract",
                                  "DOCAGENT_OCR_MODE": "regions", "DOCAGENT_OCR_LANG": "eng"})
    assert (cfg.detector_prefer, cfg.ocr_kind, cfg.ocr_mode, cfg.ocr_lang) == (
        "roboflow", "tesseract", "regions", "eng")
    assert DocAgentConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("settings", [{"ocr_kind": "other"}, {"ocr_mode": "other"}, {"ocr_lang": ""}])
def test_config_rejects_invalid_ocr(settings):
    with pytest.raises(ValueError):
        DocAgentConfig(**settings)


def test_cli_writes_report_without_llm_or_motion(blank_image, fake_tesseract, tmp_path, monkeypatch):
    image_path = tmp_path / "빈 문서.png"
    image_path.write_bytes(cv2.imencode('.png', blank_image)[1].tobytes())
    output = tmp_path / 'outputs' / 'result.json'
    monkeypatch.setattr(analyze, '_load_dotenv', lambda: None)
    monkeypatch.setenv('DOCAGENT_LLM_KIND', 'claude')
    client = WorkflowClient()
    monkeypatch.setattr(analyze, 'build_detector', lambda *a, **kw: rf.RoboflowDetector(client=client))
    assert analyze.main([str(image_path), '--detector', 'roboflow', '--dpi', '254',
                         '--output', str(output), '--allow-unmasked-blank']) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['ocr_mode'] == 'regions'
    assert len(report['fields']) == 2
    assert len(client.calls) == 1


def test_cli_preflight_fails_before_remote_inference(blank_image, tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "form.png"
    image_path.write_bytes(cv2.imencode('.png', blank_image)[1].tobytes())
    monkeypatch.setattr(analyze, '_load_dotenv', lambda: None)
    def fail(self):
        raise VisionError('언어 데이터 없음')
    monkeypatch.setattr(TesseractOcr, 'check_available', fail)
    def forbidden(*a, **kw):
        pytest.fail('remote client must not be created before OCR preflight succeeds')
    monkeypatch.setattr(analyze, 'build_detector', forbidden)
    assert analyze.main([str(image_path), '--detector', 'roboflow']) == 1
    assert '언어 데이터' in capsys.readouterr().err


def test_cli_missing_file_has_actionable_error(capsys):
    assert analyze.main(['not-a-file.png']) == 1
    assert '파일이 없습니다' in capsys.readouterr().err


def test_remote_failure_message_does_not_contain_secret():
    class FailingClient:
        def run_workflow(self, **kwargs):
            raise RuntimeError('request failed api_key=TEST_SECRET_TOKEN')
    with pytest.raises(VisionError) as error:
        rf.RoboflowDetector(client=FailingClient()).predict('image.png', allow_unmasked_blank=True)
    assert 'TEST_SECRET_TOKEN' not in str(error.value)


def test_local_analysis_never_initializes_egress_or_agent(
    blank_image, fake_tesseract, monkeypatch,
):
    # Real blank forms exposed false positives when numeric metadata was combined
    # by the LLM egress gate. Local OCR review has no LLM payload to authorize.
    import docagent.pipeline as pipeline

    def forbidden(*args, **kwargs):
        pytest.fail('local vision analysis must not initialize external-action stages')

    for name in ('LlmEgressGate', 'build_index', '_build_inner_llm',
                 'MockMotionController', 'ScriptedSpeechIO', 'DocumentAgent'):
        monkeypatch.setattr(pipeline, name, forbidden)
    result = pipeline.analyze_document(
        blank_image, detector=BlankDocumentDetector(rf.RoboflowDetector(client=WorkflowClient())),
        config=DocAgentConfig(dpi=254, ocr_kind='tesseract', ocr_mode='regions'),
    )
    assert len(result.structure.fields) == 2
    assert [stage.stage for stage in result.stages] == ['normalize', 'detect', 'ocr', 'structure']


def test_cli_requires_privacy_policy_before_constructing_api(blank_image, fake_tesseract, tmp_path, monkeypatch, capsys):
    image = tmp_path/'form.png'; image.write_bytes(cv2.imencode('.png',blank_image)[1].tobytes())
    monkeypatch.setattr(analyze,'_load_dotenv',lambda:None)
    def forbidden(*args,**kwargs): pytest.fail('must not create API client')
    monkeypatch.setattr(analyze,'build_detector',forbidden)
    assert analyze.main([str(image),'--detector','roboflow']) == 1
    assert '--mask-plan' in capsys.readouterr().err


def test_cli_sends_masked_pixels_but_ocr_stays_local(blank_image, fake_tesseract, tmp_path, monkeypatch):
    from docagent.vision.normalize import normalize_document
    from docagent.pii.image_mask import draft_plan, plan_digest
    image = tmp_path/'form.png'; image.write_bytes(cv2.imencode('.png',blank_image)[1].tobytes())
    normalized = normalize_document(blank_image,dpi=254).image
    p = draft_plan(normalized); p['boxes'] = [[50,50,100,100]]
    p['approval'] = dict(status='approved',reviewed_by='test',reviewed_at='2026-10-15',content_sha256=plan_digest(p))
    plan = tmp_path/'plan.json';plan.write_text(json.dumps(p))
    output=tmp_path/'result.json'
    monkeypatch.setattr(analyze,'_load_dotenv',lambda:None)
    client=WorkflowClient()
    monkeypatch.setattr(analyze,'build_detector',lambda *a,**kw:rf.RoboflowDetector(client=client))
    assert analyze.main([str(image),'--detector','roboflow','--dpi','254','--mask-plan',str(plan),'--output',str(output)]) == 0
    sent=client.calls[0]['images']['image']
    assert np.all(sent[50:150,50:150] == 0)
    assert not np.array_equal(sent,normalized)
    assert np.all(blank_image == 255)
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['text_included'] is False
    assert all(w['text']=='' for w in report['ocr_words'])
    assert all(f['title']=='' for f in report['fields'])
    # A changed mask cannot cause a second upload.
    p['boxes'][0][0]=51; plan.write_text(json.dumps(p))
    assert analyze.main([str(image),'--detector','roboflow','--dpi','254','--mask-plan',str(plan)]) == 1
    assert len(client.calls)==1


def test_report_text_export_is_explicit(blank_image,fake_tesseract):
    from dataclasses import replace
    from docagent.pipeline import analyze_document
    from docagent.contracts import OcrWord, BoxMm
    session=analyze_document(blank_image,detector=BlankDocumentDetector(rf.RoboflowDetector(client=WorkflowClient())),
                            config=DocAgentConfig(dpi=254,ocr_kind='tesseract',ocr_mode='regions'))
    session=replace(session,words=(OcrWord('synthetic-private-name',BoxMm(0,0,3,3),.99),))
    assert 'synthetic-private-name' not in json.dumps(analyze.build_report(session))
    assert 'synthetic-private-name' in json.dumps(analyze.build_report(session,include_text=True))
