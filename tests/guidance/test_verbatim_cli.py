import hashlib
import json

import pytest

from docagent import guide, review
from docagent.guidance.rules import FALLBACK, RuleCatalog


def source_catalog(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('검수용 원본', encoding='utf-8')
    original = '  문서 원문 전체\n동의 조건과 선택지.\n' + '누락 없이 읽을 문단.\n'*150 + '마지막 주의사항.  '
    data = dict(schema_version=1, purpose='production', fields=[], pages=[dict(
        document_id='fixture-document',document_version='1',page=1,page_count=1,reading_mode='verbatim',
        source_text=original,source=dict(reference='synthetic-fixture-only',sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        approval=review.draft_approval())])
    return data,source,original


def test_read_entire_document_cli_preserves_source_bytes_and_audio(tmp_path,monkeypatch,capsys):
    data,source,original=source_catalog(tmp_path)
    data=review.page_edit(data,'fixture-document','1',1,reviewer='synthetic-fixture',source_file=source)
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(data),encoding='utf-8')
    class Backend:
        state='completed'
        def __init__(self): self.spoken=[]
        def play(self,text): self.spoken.append(text)
        def wait(self): pass
        def stop(self): pass
    backend=Backend();monkeypatch.setattr(guide,'SayBackend',lambda **kw:backend)
    assert guide.main(['--catalog',str(catalog),'--read-document','--document-id','fixture-document',
                       '--document-version','1','--speak'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['source_text']==original and result['instruction']==original
    assert backend.spoken==[original]


def test_draft_page_cli_never_returns_partial_text(tmp_path,capsys):
    data,_,original=source_catalog(tmp_path)
    p=tmp_path/'catalog.json';p.write_text(json.dumps(data),encoding='utf-8')
    assert guide.main(['--catalog',str(p),'--read-page','--document-id','fixture-document',
                       '--document-version','1','--page','1'])==2
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='unapproved' and result['instruction']==FALLBACK
    assert original not in json.dumps(result)


def test_page_review_requires_correct_original_and_explicit_confirmation(tmp_path):
    data,source,original=source_catalog(tmp_path)
    p=tmp_path/'catalog.json';p.write_text(json.dumps(data),encoding='utf-8')
    output=tmp_path/'approved.json'
    args=['page-approve',str(p),'--document-id','fixture-document','--document-version','1',
          '--page','1','--source-file',str(source),'--reviewer','synthetic-fixture','--output',str(output)]
    with pytest.raises(SystemExit): review.main(args)
    source.write_text('원본 변경',encoding='utf-8')
    assert review.main(args+['--confirm-reviewed'])==1 and not output.exists()
    source.write_text('검수용 원본',encoding='utf-8')
    assert review.main(args+['--confirm-reviewed'])==0
    assert RuleCatalog.load(output).read_document('fixture-document','1').instruction==original
    approved=json.loads(output.read_text())
    draft=review.page_edit(approved,'fixture-document','1',1)
    assert draft['pages'][0]['source_text']==original
    assert RuleCatalog(draft).read_page('fixture-document','1',1).status=='unapproved'


@pytest.mark.parametrize('extra',[
    ['--read-page'],['--read-page','--document-id','d','--document-version','1'],
    ['--read-document','--document-id','d','--document-version','1','--page','1'],
    ['--read-document','--document-id','d','--document-version','1','--field-id','f'],
])
def test_source_selection_must_be_explicit(tmp_path,extra):
    with pytest.raises(SystemExit): guide.main(['--catalog',str(tmp_path/'not-read.json'),*extra])
