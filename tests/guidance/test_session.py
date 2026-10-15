"""검토 기록·상태 유지·안내 이동·외부 전송 순서 회귀 검사."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from docagent import run
from docagent.config import DocAgentConfig
from docagent.detector.roboflow_detector import RoboflowDetector
from docagent.errors import PiiEgressBlocked
from docagent.guidance.rules import RuleCatalog
from docagent.guidance.session import GuidanceSession, context_digest, digest
from docagent.vision.quality import CaptureQuality
from tests.guidance.test_analysis import report, context


def reviewed_context(data):
    from docagent.pii.image_mask import image_digest
    c = dict(context(), schema_version=1, analysis_sha256=digest(data), field_order=['choice_01'],
             source_image_sha256=data.get('source_image_sha256',image_digest(np.ones((10,10),np.uint8))))
    c['approval'] = dict(status='approved', reviewed_by='synthetic-reviewer', reviewed_at='2026-10-15',
                         content_sha256=context_digest(c))
    return c


def catalog():
    return RuleCatalog.load('examples/guidance/catalog.demo.json')


class Speaker:
    state = 'playing'
    def __init__(self): self.spoken = []; self.stops = 0
    def play(self, text): self.spoken.append(text)
    def stop(self): self.stops += 1
    def wait(self): pass


def test_session_exact_instruction_and_navigation():
    data = report(); speaker = Speaker()
    session = GuidanceSession(catalog(), data, reviewed_context(data), demo=True, backend=speaker)
    original = session.current()['instruction']
    for command in ('r', 'n', 'p', 's', 'q'):
        assert session.command(command)['instruction'] == original
    assert speaker.spoken == [original] * 3
    assert session.current()['position'] == 1


@pytest.mark.parametrize('change', [lambda r: r.update(needs_review=True),
    lambda r: r['fields'][0].update(needs_review=True),
    lambda r: r['stages'][0].update(ok=False),
    lambda r: r.update(capture_quality={'status':'not_checked'}),
    lambda r: r['ocr_words'][0].update(confidence=.3)])
def test_human_context_cannot_override_analysis_warnings(change):
    data = report(); change(data); speaker = Speaker()
    session = GuidanceSession(catalog(), data, reviewed_context(data), demo=True, backend=speaker)
    assert not session.ready
    assert session.command('r')['status'] == 'blocked'
    assert speaker.spoken == []


@pytest.mark.parametrize('change', [lambda c: c.update(identity_verified=False),
    lambda c: c.update(field_order=[]), lambda c: c.update(field_order=['choice_01'] * 2),
    lambda c: c.update(page=2), lambda c: c['approval'].update(status='draft')])
def test_invalid_context_stops_before_capture_or_api(change):
    data = report(); c = reviewed_context(data); change(c)
    def forbidden(*a, **kw): pytest.fail('preflight must stop')
    result, session = run.execute(catalog=catalog(), context=c, image=np.ones((10,10),np.uint8), demo=True,
                                  detector_factory=forbidden, ocr_factory=forbidden)
    assert session is None and result['reason'] == 'document_review_invalid'


def test_report_changed_after_review_is_rejected():
    data = report(); c = reviewed_context(data); data['ocr_words'][0]['text'] += '변경'
    result, session = run.execute(catalog=catalog(), context=c, report=data, demo=True)
    assert session is None and result['reason'] == 'guidance_invalid'


def test_remote_adapters_and_legacy_pipeline_fail_closed():
    client = SimpleNamespace(run_workflow=lambda **kw: pytest.fail('unreviewed upload'))
    detector = RoboflowDetector(client=client)
    for call in (lambda: detector.predict('image.png'), lambda: detector.detect(np.ones((10,10,3),np.uint8))):
        with pytest.raises(PiiEgressBlocked): call()
    from docagent.pipeline import build_session
    with pytest.raises(PiiEgressBlocked):
        build_session(np.full((100,70,3),255,np.uint8), detector=detector)


def test_quality_and_missing_mask_block_before_api(monkeypatch):
    data = report(); c = reviewed_context(data)
    def forbidden(*a, **kw): pytest.fail('API/OCR factory must not run')
    monkeypatch.setattr(run, 'assess_capture', lambda *a, **kw: CaptureQuality('retake', ('blurred',)))
    result, _ = run.execute(catalog=catalog(), context=c, image=np.ones((10,10),np.uint8), demo=True,
                           detector_factory=forbidden, ocr_factory=forbidden)
    assert result['reason'] == 'capture_quality'
    monkeypatch.setattr(run, 'assess_capture', lambda *a, **kw: CaptureQuality('pass'))
    result, _ = run.execute(catalog=catalog(), context=c, image=np.ones((10,10),np.uint8), demo=True,
                           config=DocAgentConfig(detector_prefer='roboflow'),
                           detector_factory=forbidden, ocr_factory=forbidden)
    assert result['reason'] == 'mask_required'


def test_image_to_guidance_and_cli_offline(monkeypatch, tmp_path, capsys):
    data = report(); quality = CaptureQuality('pass'); data['capture_quality'] = quality.to_dict()
    from docagent.pii.image_mask import image_digest
    data['source_image_sha256'] = image_digest(np.ones((10,10),np.uint8))
    c = reviewed_context(data)
    monkeypatch.setattr(run, 'assess_capture', lambda *a, **kw: quality)
    monkeypatch.setattr(run, 'analyze_document', lambda *a, **kw: object())
    monkeypatch.setattr(run, 'build_report', lambda *a, **kw: copy.deepcopy(data))
    result, session = run.execute(catalog=catalog(), context=c, image=np.ones((10,10),np.uint8), demo=True,
        detector_factory=lambda *a: object(), ocr_factory=lambda **kw: SimpleNamespace(check_available=lambda: None))
    assert result['status'] == 'ready' and session.ready
    a=tmp_path/'analysis.json'; a.write_text(json.dumps(data),encoding='utf-8')
    b=tmp_path/'context.json'; b.write_text(json.dumps(c),encoding='utf-8')
    monkeypatch.setattr(run, '_load_dotenv', lambda: None)
    assert run.main(['--analysis',str(a),'--document-context',str(b),
                     '--catalog','examples/guidance/catalog.demo.json','--demo']) == 0
    output=capsys.readouterr().out
    assert 'unrelated-private-text' not in output
    assert json.loads(output)['status']=='ready'


def test_unsupported_or_demo_document_never_initializes_api():
    def forbidden(*a, **kw): pytest.fail('unsupported document must not upload')
    data=report(); c=reviewed_context(data)
    result,_=run.execute(catalog=catalog(),context=c,image=np.ones((10,10),np.uint8),
                         detector_factory=forbidden,ocr_factory=forbidden)
    assert result['reason']=='unsupported_or_demo_document'
    c['document_id']='unsupported';c['approval']['content_sha256']=context_digest(c)
    result,_=run.execute(catalog=catalog(),context=c,image=np.ones((10,10),np.uint8),demo=True,
                         detector_factory=forbidden,ocr_factory=forbidden)
    assert result['reason']=='unsupported_or_demo_document'


def test_changed_source_image_stops_before_quality_ocr_or_upload(monkeypatch):
    def forbidden(*a, **kw): pytest.fail('changed source must not proceed')
    monkeypatch.setattr(run,'assess_capture',forbidden)
    result,_=run.execute(catalog=catalog(),context=reviewed_context(report()),image=np.zeros((10,10),np.uint8),
                         demo=True,detector_factory=forbidden,ocr_factory=forbidden)
    assert result['reason']=='source_image_changed'
