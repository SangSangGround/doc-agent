"""실제 정확도를 대신하지 않는 체크 결과·정합 거절 회귀 검사."""
import copy
import json

import cv2
import numpy as np
import pytest

from docagent.vision.checkbox_review import verify_checkbox_groups


GROUPS = [{"id": "consent", "source_text": "동의 여부", "min_selected": 1, "max_selected": 1,
           "options": [{"id": "yes", "label": "동의함", "box_mm": [42, 80, 6, 6]},
                       {"id": "no", "label": "동의하지 않음", "box_mm": [114, 80, 6, 6]}]}]


@pytest.fixture(scope="module")
def page():
    image = np.full((1188, 840), 255, np.uint8)
    for row, y in enumerate(range(65, 1140, 48)):
        if 290 < y < 365:
            continue
        cv2.putText(image, f"FORM {row:02d} reference {row*731+234} SAMPLE", (55, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, 0, 2)
    for option in GROUPS[0]["options"]:
        x, y, w, h = (round(v*4) for v in option["box_mm"])
        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), 0, 2)
    return image


def mark(page, option_id="yes", color=0):
    after = page.copy()
    option = next(o for o in GROUPS[0]["options"] if o["id"] == option_id)
    x, y, _, _ = (round(v*4) for v in option["box_mm"])
    cv2.polylines(after, [np.array([[x+5,y+12], [x+10,y+18], [x+19,y+5]], np.int32)],
                  False, color, 2)
    return after


def review(page, after, **kwargs):
    return verify_checkbox_groups(page, after, GROUPS, capture_mode="scan", **kwargs)


def translated(image, dx, dy):
    height, width = image.shape[:2]
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (width, height), borderValue=(255, 255, 255))


@pytest.mark.parametrize("ids,expected", [([], "unmarked"), (["yes"], "single"),
                                         (["no"], "single"), (["yes", "no"], "multiple")])
def test_observed_options_are_not_collapsed_to_any_marked(page, ids, expected):
    after = page.copy()
    for option_id in ids:
        after = mark(after, option_id)
    result = review(page, after)
    assert result["status"] == "ok", result
    assert result["groups"][0]["state"] == expected
    assert result["groups"][0]["marked_option_ids"] == ids
    json.dumps(result, allow_nan=False)


def test_scan_translation_is_registered_before_verification(page):
    written = mark(page, "no")
    shifted = translated(written, 9, 11)
    result = review(page, shifted)
    assert result["status"] == "ok", result
    assert result["groups"][0]["marked_option_ids"] == ["no"]


def test_jpeg_capture_and_uniform_illumination_preserve_observed_option(page):
    written = (mark(page, "no").astype(float)*.8).astype(np.uint8)
    encoded = cv2.imencode('.jpg', written, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
    captured = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    result = review(page, captured)
    assert result["status"] == "ok", result
    assert result["groups"][0]["marked_option_ids"] == ["no"]


def test_document_photo_with_background_is_rectified(page):
    canvas = np.full((1440, 1080), 65, np.uint8)
    canvas[126:1314,120:960] = mark(page)
    result = verify_checkbox_groups(page, canvas, GROUPS)
    assert result["status"] == "ok", result
    assert result["groups"][0]["marked_option_ids"] == ["yes"]


@pytest.mark.parametrize("changed", ["faint", "gray_cover", "filled", "dot"])
def test_unclear_marks_and_occlusions_are_not_reported_complete(page, changed):
    after = page.copy()
    if changed == "faint":
        after = mark(page, color=215)
    elif changed == "dot":
        cv2.circle(after, (180,332), 1, 0, -1)
    else:
        after[325:339,173:187] = 160 if changed == "gray_cover" else 0
    result = review(page, after)
    assert result["status"] == "uncertain", result
    assert result["groups"][0]["state"] == "uncertain"


def test_blurry_image_and_missing_page_never_become_unmarked(page):
    for after in [cv2.GaussianBlur(page, (31,31), 7), page[130:].copy(), np.full_like(page, 255)]:
        result = review(page, after)
        assert result["status"] == "uncertain", result
        assert result["groups"][0]["state"] == "uncertain"


def test_same_dimensions_are_not_evidence_of_same_document(page):
    other = page.copy()
    other[400:1100] = 255
    for y in range(440,1080,50):
        cv2.putText(other, "OTHER DOCUMENT 98765432", (70,y), cv2.FONT_HERSHEY_SIMPLEX,.8,0,2)
    result = review(page, other)
    assert result["status"] == "uncertain", result
    assert result["reason"] == "alignment_failed"


def test_reference_with_existing_mark_is_rejected(page):
    reference = mark(page)
    result = review(reference, mark(reference, "no"))
    assert result["status"] == "uncertain", result
    assert result["groups"][0]["options"][0]["reason"] == "reference_not_blank"


@pytest.mark.parametrize("color", [215, 235, 245])
def test_faint_strokes_request_review_instead_of_unmarked(page, color):
    result = review(page, mark(page, color=color))
    assert result["groups"][0]["options"][0]["state"] == "uncertain", result


def test_white_occlusion_cannot_hide_a_checkbox_as_unmarked(page):
    covered = mark(page)
    covered[315:350,160:200] = 255
    result = review(page, covered)
    assert result["groups"][0]["state"] == "uncertain", result
    assert result["groups"][0]["options"][0]["reason"] == "checkbox_border_obscured"


def test_extra_tick_near_unselected_box_prevents_successful_observation(page):
    after = mark(page, "no")
    cv2.polylines(after, [np.array([[152,328],[160,340],[164,320]],np.int32)],False,0,2)
    result = review(page, after)
    assert result["status"] == "uncertain", result
    assert result["groups"][0]["options"][0]["reason"] == "mark_outside_checkbox"


@pytest.fixture(scope="module")
def home_demo(tmp_path_factory):
    from docagent.home.demo import create_demo
    path = tmp_path_factory.mktemp("checkbox-home-demo")
    manifest = json.loads(create_demo(path).read_text(encoding="utf-8"))
    return path, manifest, cv2.imread(str(path/"reference.png"))


@pytest.mark.parametrize("name,first_state,first_ids", [
    ("correct","single",["disagree"]), ("opposite","single",["agree"]),
    ("missing","unmarked",[]), ("duplicate","multiple",["agree","disagree"]),
])
def test_korean_demo_has_expected_per_group_states(home_demo, name, first_state, first_ids):
    path, manifest, reference = home_demo
    result = verify_checkbox_groups(reference, cv2.imread(str(path/f"{name}.png")),
                                    manifest["groups"], capture_mode="scan")
    assert result["status"] == "ok", result
    assert result["groups"][0]["state"] == first_state
    assert result["groups"][0]["marked_option_ids"] == first_ids
    assert result["groups"][1]["marked_option_ids"] == ["disagree"]


@pytest.mark.parametrize("transform", [
    pytest.param((10, 12), id="translate"),
    pytest.param((9, 4), id="translate-down-right"),
    pytest.param((-7, -1), id="translate-up-left"),
    "rotate", "lighting",
])
def test_korean_demo_capture_variation_preserves_marked_cells(home_demo, transform):
    path, manifest, reference = home_demo
    captured = cv2.imread(str(path/"correct.png"))
    h,w = captured.shape[:2]
    if isinstance(transform, tuple):
        captured = translated(captured, *transform)
    elif transform == "rotate":
        captured = cv2.warpAffine(captured,cv2.getRotationMatrix2D((w/2,h/2),2,1),(w,h),borderValue=(255,255,255))
    else:
        captured = np.clip(captured*np.linspace(.55,1,h)[:,None,None],0,255).astype(np.uint8)
    result = verify_checkbox_groups(reference,captured,manifest["groups"],capture_mode="scan")
    assert result["status"] == "ok", (result["alignment"].get("reason"), result["alignment"])
    assert all(group["marked_option_ids"] == ["disagree"] for group in result["groups"])


@pytest.mark.parametrize("obstruction", ["white", "gray", "header", "stray", "erasure"])
def test_korean_demo_damaged_capture_is_uncertain(home_demo, obstruction):
    path, manifest, reference = home_demo
    captured = cv2.imread(str(path/"correct.png"))
    if obstruction in ("white", "gray"):
        captured[708:754,592:646] = 255 if obstruction == "white" else 160
    elif obstruction == "header":
        captured[15:400] = 255
    elif obstruction == "erasure":
        # 반대 칸의 지워지지 않은 옅은 흔적을 미표시로 통과시키지 않는다.
        cv2.polylines(captured,[np.array([[155,732],[166,744],[182,719]],np.int32)],False,(235,235,235),3)
    else:
        cv2.polylines(captured,[np.array([[126,726],[138,744],[144,714]],np.int32)],False,(0,0,0),3)
    result = verify_checkbox_groups(reference,captured,manifest["groups"],capture_mode="scan")
    assert result["status"] == "uncertain", result


@pytest.mark.parametrize("side", ["left", "right", "bottom"])
@pytest.mark.parametrize("color", [0, 235])
def test_margin_only_marks_are_not_lost_by_checkbox_inset(home_demo,side,color):
    path,manifest,reference = home_demo
    captured = cv2.imread(str(path/"correct.png"))
    x,y,w,h = [round(v*6) for v in manifest["groups"][0]["options"][0]["box_mm"]]
    points = {"left":((x+4,y+8),(x+4,y+27)),
              "right":((x+w-4,y+8),(x+w-4,y+27)),
              "bottom":((x+8,y+h-4),(x+27,y+h-4))}
    cv2.line(captured,*points[side],(color,color,color),2)
    result = verify_checkbox_groups(reference,captured,manifest["groups"],capture_mode="scan")
    assert result["status"] == "uncertain", result
    assert result["groups"][0]["options"][0]["reason"] == "mark_at_checkbox_margin"


@pytest.mark.parametrize("shift", [(0, 0), (9, 4)], ids=["unshifted", "translated"])
def test_small_missing_printed_phrase_is_not_hidden_by_global_similarity(home_demo, shift):
    path,manifest,reference = home_demo
    captured = cv2.imread(str(path/"correct.png"))
    captured[535:555,108:230] = 255
    if shift != (0, 0):
        captured = translated(captured, *shift)
    result = verify_checkbox_groups(reference,captured,manifest["groups"],capture_mode="scan")
    assert result["status"] == "uncertain", result
    assert result["alignment"]["reason"] == "local_printed_content_mismatch"


@pytest.mark.parametrize("box", [[-1,80,6,6], [209,80,6,6], [42,80,0,6],
                                  [42,80,float("nan"),6], [True,80,6,6]])
def test_invalid_boxes_fail_before_processing(page, box):
    groups = copy.deepcopy(GROUPS)
    groups[0]["options"][0]["box_mm"] = box
    with pytest.raises(ValueError):
        verify_checkbox_groups(page, page, groups)


def test_overlapping_options_cannot_count_one_mark_twice(page):
    groups = copy.deepcopy(GROUPS)
    groups[0]["options"][1]["box_mm"] = [43,81,6,6]
    with pytest.raises(ValueError):
        verify_checkbox_groups(page, page, groups)


@pytest.mark.parametrize("image", [None, np.array([]), np.zeros((12,12), float),
                                    np.zeros((800,800,4),np.uint8)])
def test_invalid_capture_returns_uncertain(page, image):
    assert review(page, image)["reason"] == "invalid_image"


def test_blank_featureless_reference_is_not_registered_by_dimensions():
    image = np.full((1188,840),255,np.uint8)
    result = review(image, image)
    assert result["status"] == "uncertain"


def test_inputs_are_not_mutated(page):
    original = page.copy()
    after = mark(page)
    written = after.copy()
    review(page, after)
    np.testing.assert_array_equal(page, original)
    np.testing.assert_array_equal(after, written)
