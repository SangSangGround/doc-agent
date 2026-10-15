import json
from docagent.testing.workflow import generate
from docagent.evaluate import evaluate_manifest
from docagent.guidance.rules import RuleCatalog
from docagent.guidance.session import GuidanceSession
from tests.guidance.test_session import Speaker


def test_reproducible_fixture_navigation_and_evaluation(tmp_path):
    out=tmp_path/'demo';generate(out)
    load=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    session=GuidanceSession(RuleCatalog(load('catalog.json')),load('analysis.json'),load('context.json'),demo=True,backend=Speaker())
    assert session.ready
    first=session.command('r');second=session.command('n');back=session.command('p')
    assert first['field_id']=='participation' and second['field_id']=='applicant-signature'
    assert back==first and session.backend.spoken==[first['instruction'],second['instruction'],first['instruction']]
    report=evaluate_manifest(out/'manifest.json',demo=True)
    assert report['by_variant']['original']['detection']['classes']['signature']['recall']==1
    assert report['by_variant']['photo']['detection']['classes']['signature']['recall']==0
    assert report['overall']['guidance']['missed_match']==1
    assert report['audit']['unknown_training']==['sample_0001','sample_0002']
