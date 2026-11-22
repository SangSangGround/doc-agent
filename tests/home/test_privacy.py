import json

import numpy as np
import pytest

from docagent.contracts import BoxMm, OcrWord
from docagent.home.privacy import propose_redactions, render_redaction_preview


class LocalWords:
    page_size_mm = (800., 1000.)

    def __init__(self, words):
        self.words = words

    def read(self, image):
        return self.words


def word(text, x, y=100., width=90., confidence=.95):
    return OcrWord(text, BoxMm(float(x), float(y), float(width), 20.), confidence)


def picture():
    return np.full((1000, 800, 3), 255, np.uint8)


def test_split_phone_and_rrn_map_every_part_without_returning_original_text():
    words = [word('전화번호', 10), word('010', 110), word('-', 210, width=10),
             word('1234', 230), word('-', 330, width=10), word('5678', 350),
             word('주민등록번호', 10, y=150), word('900101', 110, y=150),
             word('1234567', 250, y=150)]
    result = propose_redactions(picture(), ocr=LocalWords(words))
    assert result['status'] == 'draft'
    assert result['candidate_count'] >= 2
    rectangles = [row['box_px'] for row in result['boxes']]
    for index in (1, 3, 5, 7, 8):
        assert [int(words[index].box_mm.x_mm), int(words[index].box_mm.y_mm), 90, 20] in rectangles
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    for secret in ('010', '1234', '5678', '900101', '1234567', '전화번호', '주민등록번호'):
        # Digest can contain digit triples; inspect values other than the opaque image hash.
        public = dict(result, image_sha256=None)
        assert secret not in json.dumps(public, ensure_ascii=False)
    assert not result['coverage_verified']
    assert not result['external_transfer_allowed']
    assert result['requires_review']


def test_unknown_names_are_not_misreported_as_safe():
    result = propose_redactions(picture(), ocr=LocalWords([word('홍길동', 100)]))
    assert result['status'] == 'draft'
    assert result['boxes'] == []
    assert 'unlabelled_names_not_verified' in result['warnings']
    assert 'no_candidates_does_not_mean_safe' in result['warnings']
    assert not result['coverage_verified']
    assert not result['external_transfer_allowed']


def test_low_confidence_words_are_included_and_reported_for_review():
    words = [word('연락처', 10), word('010-1234-5678', 110, width=180, confidence=.2)]
    result = propose_redactions(picture(), ocr=LocalWords(words))
    assert result['boxes']
    assert 'low_ocr_confidence' in result['warnings']


def test_empty_ocr_and_adapter_failure_block_without_echoing_values():
    assert propose_redactions(picture(), ocr=LocalWords([]))['reason'] == 'ocr_no_text'

    class Broken:
        def read(self, image):
            raise RuntimeError('raw private data 900101-1234567')

    result = propose_redactions(picture(), ocr=Broken())
    assert result['status'] == 'blocked'
    assert result['reason'] == 'ocr_or_detection_failed'
    assert '900101' not in json.dumps(result)
    assert not result['external_transfer_allowed']


def test_out_of_bounds_ocr_fails_entire_draft():
    words = [word('010-1234-5678', 100), word('불확실', 799)]
    result = propose_redactions(picture(), ocr=LocalWords(words))
    assert result['status'] == 'blocked'
    assert result['boxes'] == []


@pytest.mark.parametrize('invalid', [None, np.zeros((10, 10), float), np.zeros((10, 10, 4), np.uint8)])
def test_invalid_image_does_not_call_ocr(invalid):
    class Forbidden:
        def read(self, image):
            pytest.fail('invalid image must not reach OCR')
    assert propose_redactions(invalid, ocr=Forbidden())['status'] == 'blocked'


def test_preview_changes_only_copy_and_cannot_reuse_other_image():
    original = picture()
    result = propose_redactions(original, ocr=LocalWords([word('010-1234-5678', 100, width=180)]))
    masked = render_redaction_preview(original, result)
    assert np.all(original == 255)
    assert np.all(masked[100:120, 100:280] == 0)
    assert np.all(masked[:90, :] == 255)
    other = original.copy()
    other[0, 0] = 0
    with pytest.raises(ValueError):
        render_redaction_preview(other, result)
    with pytest.raises(ValueError):
        render_redaction_preview(original, dict(result, external_transfer_allowed=True))


def test_zero_candidate_preview_remains_unapproved_copy():
    original = picture()
    result = propose_redactions(original, ocr=LocalWords([word('안내문', 100)]))
    masked = render_redaction_preview(original, result)
    assert np.array_equal(original, masked)
    assert masked is not original
    assert not result['external_transfer_allowed']


def test_bad_preview_boxes_are_rejected():
    image = picture()
    result = propose_redactions(image, ocr=LocalWords([word('안내문', 100)]))
    result['boxes'] = [{'box_px': [799, 10, 20, 20]}]
    with pytest.raises(ValueError):
        render_redaction_preview(image, result)
