"""OCR 설정이 파일·환경·CLI와 실제 어댑터까지 일관되게 전달된다."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from docagent import analyze, run
from docagent.config import DocAgentConfig
from docagent.contracts import BoxMm, Detection, FieldType
from docagent.pipeline import analyze_document
from docagent.vision.ocr import TesseractOcr


@pytest.mark.parametrize("psm", [None, 3, 6, 11])
def test_ocr_settings_roundtrip_and_existing_defaults(psm):
    cfg = DocAgentConfig(ocr_mode="regions", ocr_psm=psm, ocr_preprocess="form_lines")
    assert DocAgentConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.tesseract_config() == f"--psm {11 if psm is None else psm}"
    assert DocAgentConfig.from_dict({}).ocr_psm is None
    assert DocAgentConfig().tesseract_config() == ""


@pytest.mark.parametrize("psm", [True, False, 6.0, "6", 0, 5, 12])
def test_invalid_psm_is_not_silently_coerced(psm):
    with pytest.raises(ValueError, match="ocr_psm"):
        DocAgentConfig(ocr_psm=psm)
    with pytest.raises(ValueError, match="ocr_psm"):
        DocAgentConfig.from_dict({"ocr_psm": psm})


def test_environment_selects_psm_and_preprocessing_together():
    cfg = DocAgentConfig.from_env({"DOCAGENT_OCR_PSM": "6", "DOCAGENT_OCR_MODE": "regions",
                                   "DOCAGENT_OCR_PREPROCESS": "form_lines"})
    assert cfg.ocr_psm == 6
    assert cfg.ocr_preprocess == "form_lines"
    assert cfg.tesseract_config() == "--psm 6"
    assert DocAgentConfig.from_env({}, base=cfg) == cfg


@pytest.mark.parametrize("value", ["6.5", "automatic", "5"])
def test_environment_rejects_invalid_psm(value):
    with pytest.raises(ValueError):
        DocAgentConfig.from_env({"DOCAGENT_OCR_PSM": value})


def test_page_form_lines_is_blocked_by_all_config_paths():
    with pytest.raises(ValueError, match="regions"):
        DocAgentConfig(ocr_preprocess="form_lines")
    with pytest.raises(ValueError, match="regions"):
        DocAgentConfig.from_dict({"ocr_mode": "page", "ocr_preprocess": "form_lines"})
    with pytest.raises(ValueError, match="regions"):
        DocAgentConfig.from_env({"DOCAGENT_OCR_PREPROCESS": "form_lines"})


class Detector:
    def detect(self, image):
        return [Detection(FieldType.CHECKBOX, BoxMm(60, 100, 5, 5), .93, "synthetic-test")]


@pytest.fixture
def fake_ocr(monkeypatch):
    calls = []

    def image_to_data(image, **kwargs):
        calls.append(kwargs)
        return dict(text=["선택"], conf=[33], left=[520], top=[150], width=[100], height=[40])

    sdk = SimpleNamespace(image_to_data=image_to_data, Output=SimpleNamespace(DICT="dict"),
                          get_languages=lambda **kw: ["kor", "eng"])
    monkeypatch.setattr(TesseractOcr, "_import_pytesseract", staticmethod(lambda: sdk))
    return calls


@pytest.mark.parametrize("psm,preprocess,expected_calls", [(None, "none", 1), (6, "form_lines", 2)])
def test_pipeline_uses_selected_psm_and_preserves_both_evidence(psm, preprocess, expected_calls, fake_ocr):
    result = analyze_document(
        np.full((594, 420), 255, np.uint8), detector=Detector(),
        config=DocAgentConfig(dpi=254, ocr_kind="tesseract", ocr_mode="regions",
                              ocr_psm=psm, ocr_preprocess=preprocess),
    )
    assert len(fake_ocr) == expected_calls
    assert all(call["config"] == f"--psm {11 if psm is None else psm}" for call in fake_ocr)
    assert result.words[0].confidence == .33
    if preprocess == "form_lines":
        assert result.ocr_evidence["baseline_words"][0]["confidence"] == .33
        assert result.ocr_evidence["words"][0]["box_mm"] == result.words[0].box_mm.to_dict()


def test_analyze_cli_psm_overrides_environment_and_records_settings(tmp_path, monkeypatch, fake_ocr):
    source = tmp_path / "form.png"
    source.write_bytes(cv2.imencode(".png", np.full((594, 420), 255, np.uint8))[1].tobytes())
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(analyze, "_load_dotenv", lambda: None)
    monkeypatch.setattr(analyze, "build_detector", lambda *a, **kw: Detector())
    monkeypatch.setenv("DOCAGENT_OCR_PSM", "11")
    assert analyze.main([str(source), "--detector", "heuristic", "--dpi", "254",
                         "--ocr-psm", "6", "--ocr-preprocess", "form_lines",
                         "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ocr_psm"] == 6
    assert report["ocr_preprocess"] == "form_lines"
    assert len(fake_ocr) == 2
    assert all(call["config"] == "--psm 6" for call in fake_ocr)
    assert "ocr_evidence" not in report


def test_invalid_page_preprocess_stops_analyze_before_adapter_creation(tmp_path, monkeypatch, capsys):
    source = tmp_path / "form.png"
    source.write_bytes(cv2.imencode(".png", np.full((594, 420), 255, np.uint8))[1].tobytes())
    monkeypatch.setattr(analyze, "_load_dotenv", lambda: None)
    monkeypatch.setattr(analyze, "TesseractOcr", lambda **kw: pytest.fail("must validate first"))
    assert analyze.main([str(source), "--ocr-mode", "page", "--ocr-preprocess", "form_lines"]) == 1
    assert "regions" in capsys.readouterr().err


def test_run_cli_forwards_psm_without_executing_inference(tmp_path, monkeypatch):
    data = tmp_path / "analysis.json"
    context = tmp_path / "context.json"
    data.write_text("{}", encoding="utf-8")
    context.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run, "_load_dotenv", lambda: None)
    captured = []

    def execute(**kwargs):
        captured.append(kwargs["config"])
        return {"status": "blocked", "reason": "synthetic-test"}, None

    monkeypatch.setattr(run, "execute", execute)
    assert run.main(["--analysis", str(data), "--document-context", str(context),
                     "--catalog", "examples/guidance/catalog.demo.json", "--ocr-psm", "6",
                     "--ocr-preprocess", "form_lines"]) == 2
    assert captured[0].ocr_psm == 6
    assert captured[0].ocr_preprocess == "form_lines"
