"""지원 문서가 아닌 합성 입력으로 검수·안내·평가 명령을 재현한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from docagent.guidance.rules import content_digest
from docagent.guidance.session import context_digest
from docagent.pii.image_mask import image_digest
from docagent.review import context_draft, render_review


def generate(output: Path) -> None:
    """새 디렉터리에만 가상 자료를 생성. 승인은 합성 픽스처에만 적용한다."""
    output.mkdir(parents=True,exist_ok=False)
    def write(name: str, data: dict) -> None:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def approved(fingerprint: str) -> dict:
        return dict(status='approved',reviewed_by='synthetic-fixture-only',reviewed_at='2026-10-15',content_sha256=fingerprint)
    image=np.full((850,600,3),255,np.uint8)
    cv2.putText(image,'SYNTHETIC TEST ONLY',(35,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.rectangle(image,(171,286),(186,301),(0,0,0),2)
    cv2.putText(image,'Survey: agree / decline',(200,300),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    cv2.line(image,(171,566),(300,566),(0,0,0),2)
    cv2.putText(image,'Applicant signature',(171,550),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    for name,array in [('original.png',image),('photo.png',cv2.GaussianBlur(image,(3,3),.5))]:
        (output/name).write_bytes(cv2.imencode('.png',array)[1].tobytes())
    entries=[]
    for field_id,kind,text,instruction in [
        ('participation','checkbox','가상 설문 참여 동의함 거부함','이것은 가상 문서 테스트입니다. 설문 참여 여부를 동의함 또는 거부함 중에서 선택하세요.'),
        ('applicant-signature','signature','가상 신청인 서명','이것은 가상 문서 테스트입니다. 신청인 서명 영역입니다.')]:
        e=dict(document_id='fictional-survey',document_version='demo-1',page=1,field_id=field_id,field_type=kind,
               surrounding_text=text,keywords=text.split(),forbidden_keywords=[],instruction=instruction)
        e['approval']=approved(content_digest(e));entries.append(e)
    write('catalog.json',dict(schema_version=1,purpose='demo',fields=entries))
    b1=dict(x_mm=60,y_mm=100,w_mm=5,h_mm=5);b2=dict(x_mm=60,y_mm=190,w_mm=45,h_mm=8)
    report=dict(document_id='synthetic-only',coordinate_system='normalized_a4_mm',text_included=True,
                image_sha256=image_digest(image),source_image_sha256=image_digest(image),normalized_image_shape=list(image.shape),
                source_to_normalized=np.eye(3).tolist(),needs_review=False,capture_quality={'status':'pass'},
                fields=[dict(id='choice_01',type='choice',box_mm=b1,confidence=.99,needs_review=False),
                        dict(id='signature_01',type='signature',box_mm=b2,confidence=.99,needs_review=False)],
                detections=[dict(type='checkbox',box_mm=b1,confidence=.99),dict(type='signature',box_mm=b2,confidence=.99)],
                ocr_words=[dict(text=entries[0]['surrounding_text'],box_mm=dict(x_mm=70,y_mm=100,w_mm=40,h_mm=4),confidence=.99),
                           dict(text=entries[1]['surrounding_text'],box_mm=dict(x_mm=60,y_mm=182,w_mm=40,h_mm=4),confidence=.99)],
                stages=[dict(stage='normalize',confidence=.99,ok=True),dict(stage='ocr',confidence=.99,ok=True)])
    write('analysis.json',report)
    context=context_draft(report);context.update(document_id='fictional-survey',document_version='demo-1',identity_verified=True,confidence=.99)
    context['approval']=approved(context_digest(context));write('context.json',context)
    (output/'review.html').write_text(render_review(image,report=report),encoding='utf-8')
    from docagent.guidance.analysis import observation_from_analysis
    from dataclasses import asdict
    observed=asdict(observation_from_analysis(report,context,'choice_01'))
    expected={k:entries[0][k] for k in ('document_id','document_version','page','field_id')}
    boxes=[[60*600/210,100*850/297,5*600/210,5*850/297],[60*600/210,190*850/297,45*600/210,8*850/297]]
    coco=dict(images=[dict(id=1,file_name='original.png',width=600,height=850),dict(id=2,file_name='photo.png',width=600,height=850)],
              categories=[dict(id=10,name='check_box'),dict(id=20,name='signature_field')],annotations=[])
    samples=[]
    for i,variant in enumerate(('original','photo'),1):
        for j,b in enumerate(boxes):
            coco['annotations'].append(dict(id=(i-1)*2+j+1,image_id=i,category_id=10 if j==0 else 20,bbox=b))
        pixels=cv2.imdecode(np.frombuffer((output/f'{variant}.png').read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        predictions=[dict(**{'class':'checkbox'},bbox=boxes[0],confidence=.99)]
        if variant=='original': predictions.append(dict(**{'class':'signature'},bbox=boxes[1],confidence=.99))
        else: predictions.append(dict(**{'class':'checkbox'},bbox=[20,20,10,10],confidence=.8))
        write(f'{variant}.predictions.json',dict(coordinate_system='image_px',image_size=[600,850],image_sha256=image_digest(pixels),predictions=predictions))
        sample=dict(id=variant,image=f'{variant}.png',image_size=[600,850],variant=variant,split='external',
                    form_id='synthetic-form',source_id='synthetic-page-1',training_membership='unknown',
                    labels=dict(format='coco',path='labels.coco.json',image_id=i),predictions=f'{variant}.predictions.json',
                    guidance_cases=[dict(observation=observed if i==1 else observed|{'needs_review':True},expected=expected)],
                    mask=dict(truth=[[30,400,100,20]],predicted=[[30,400,100,20]] if i==1 else [[30,400,50,30]]))
        samples.append(sample)
    write('labels.coco.json',coco)
    write('manifest.json',dict(schema_version=1,dataset_version='synthetic-1',model_version='fake-detector',prediction_version='fixture-1',
                               catalog='catalog.json',samples=samples))


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='가상 자료만 생성합니다. 실제 승인·모델 성능을 의미하지 않습니다.')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        generate(args.output_dir)
    except (OSError,ValueError):
        print('새 출력 디렉터리를 지정하세요. 기존 파일은 덮어쓰지 않습니다.')
        return 1
    print('합성 검수·안내·평가 자료를 생성했습니다. 실제 문서에 사용하지 마세요.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
