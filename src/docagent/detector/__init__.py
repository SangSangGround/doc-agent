"""외부 추론 서비스 기반 탐지기 어댑터 패키지.

현재 구현: :mod:`docagent.detector.roboflow_detector` 의
:class:`~docagent.detector.roboflow_detector.RoboflowDetector`
(Roboflow Workflow 로 서빙되는 RF-DETR 모델).

``python -m docagent.detector.roboflow_detector`` 로 CLI 를 실행할 수 있도록
이 패키지는 하위 모듈을 미리 import 하지 않는다.
"""
