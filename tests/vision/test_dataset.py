"""라벨 데이터셋 유틸(:mod:`docagent.vision.dataset`) 테스트.

핵심은 **왕복 변환 동일성**이다.

* ``YoloLabel.from_line(label.to_line()) == label``
* ``BoxMm → YoloLabel → BoxMm`` 오차 0.01mm 이내
* ``BoxPx → YoloLabel → BoxPx`` 오차 1px 이내
* ``DocumentStructure → 라벨 파일 → 라벨 목록`` 무손실
* ``YOLO 라벨 → COCO → YOLO`` 무손실

분할·검증·yaml 기록도 결정론과 오류 보고 관점에서 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    DocumentStructure,
    Field,
    FieldType,
    Option,
    Sensitivity,
)
from docagent.vision.dataset import (
    CLASS_MAP,
    CLASS_NAMES,
    DEFAULT_SPLIT_RATIOS,
    FIELD_TYPE_TO_CLASS_ID,
    NAME_TO_CLASS_ID,
    DatasetSplit,
    LabelReport,
    YoloLabel,
    coco_to_yolo,
    export_structure_labels,
    read_yolo_labels,
    split_dataset,
    structure_to_yolo_labels,
    validate_label_dir,
    validate_label_file,
    validate_labels,
    write_data_yaml,
    write_yolo_labels,
)

A4_PAGE = (A4_WIDTH_MM, A4_HEIGHT_MM)


# --------------------------------------------------------------------------
# 클래스 정의
# --------------------------------------------------------------------------


class TestClassDefinitions:
    """클래스 매핑이 모듈 간에 일관되는지 확인한다."""

    def test_class_map_is_fixed(self) -> None:
        """클래스 id 와 이름이 고정되어 있다(학습·추론이 공유하는 계약)."""
        assert CLASS_MAP == {0: "signature_field", 1: "checkbox"}
        assert CLASS_NAMES == ("signature_field", "checkbox")

    def test_name_lookup_is_inverse(self) -> None:
        """이름 → id 매핑이 id → 이름의 역방향이다."""
        assert NAME_TO_CLASS_ID == {name: key for key, name in CLASS_MAP.items()}

    def test_field_type_mapping(self) -> None:
        """도메인 FieldType 과 클래스 id 가 1:1 로 대응한다."""
        assert FIELD_TYPE_TO_CLASS_ID == {
            FieldType.SIGNATURE: 0,
            FieldType.CHECKBOX: 1,
        }

    def test_yolo_adapter_shares_mapping(self) -> None:
        """YOLO 어댑터가 같은 매핑을 역방향으로 쓴다(ultralytics 없이 import 가능)."""
        from docagent.vision.yolo_adapter import CLASS_ID_TO_FIELD_TYPE

        assert CLASS_ID_TO_FIELD_TYPE == {0: FieldType.SIGNATURE, 1: FieldType.CHECKBOX}


# --------------------------------------------------------------------------
# YoloLabel 왕복
# --------------------------------------------------------------------------


class TestYoloLabelRoundTrip:
    """라벨 한 건의 왕복 변환 동일성."""

    @pytest.mark.parametrize(
        "label",
        [
            YoloLabel(0, 0.5, 0.5, 0.25, 0.125),
            YoloLabel(1, 0.123456, 0.987654, 0.001, 0.002),
            YoloLabel(1, 0.5, 0.5, 1.0, 1.0),
        ],
    )
    def test_line_round_trip_is_lossless(self, label: YoloLabel) -> None:
        """``from_line(to_line())`` 이 원본과 정확히 같다."""
        assert YoloLabel.from_line(label.to_line()) == label

    def test_line_format(self) -> None:
        """줄 형식은 ``class_id cx cy w h`` 다섯 값이다."""
        line = YoloLabel(1, 0.5, 0.25, 0.1, 0.2).to_line()
        tokens = line.split()
        assert len(tokens) == 5
        assert tokens[0] == "1"
        assert tokens[1] == "0.500000"

    def test_precision_is_normalized_on_construction(self) -> None:
        """생성 시점에 6자리로 반올림되므로 왕복이 무손실이 된다."""
        label = YoloLabel(0, 0.123456789, 0.5, 0.5, 0.5)
        assert label.cx == 0.123457
        assert YoloLabel.from_line(label.to_line()) == label

    def test_box_mm_round_trip(self) -> None:
        """mm 상자 → 라벨 → mm 상자 왕복 오차가 0.01mm 이내다."""
        original = BoxMm(26.0, 139.0, 6.0, 6.0)
        label = YoloLabel.from_box_mm(1, original, A4_PAGE)
        restored = label.to_box_mm(A4_PAGE)
        assert restored.x_mm == pytest.approx(original.x_mm, abs=0.01)
        assert restored.y_mm == pytest.approx(original.y_mm, abs=0.01)
        assert restored.w_mm == pytest.approx(original.w_mm, abs=0.01)
        assert restored.h_mm == pytest.approx(original.h_mm, abs=0.01)

    def test_box_px_round_trip(self) -> None:
        """픽셀 상자 → 라벨 → 픽셀 상자 왕복 오차가 1px 이내다."""
        original = BoxPx(204, 1094, 47, 47)
        label = YoloLabel.from_box_px(1, original, 1654, 2339)
        restored = label.to_box_px(1654, 2339)
        for got, want in zip(restored.to_tuple(), original.to_tuple()):
            assert abs(got - want) <= 1

    def test_class_helpers(self) -> None:
        """클래스 이름·FieldType 접근자가 매핑과 일치한다."""
        label = YoloLabel(0, 0.5, 0.5, 0.2, 0.1)
        assert label.class_name == "signature_field"
        assert label.field_type is FieldType.SIGNATURE

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"class_id": 7}, "정의되지 않은 클래스"),
            ({"cx": 1.5}, "cx"),
            ({"cy": -0.1}, "cy"),
            ({"w": 0.0}, "w"),
            ({"h": 1.5}, "h"),
        ],
    )
    def test_invalid_values_raise(self, kwargs: dict, message: str) -> None:
        """범위를 벗어난 값은 한국어 ValueError 로 막는다."""
        base = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            YoloLabel(**base)  # type: ignore[arg-type]

    @pytest.mark.parametrize("line", ["0 0.5 0.5 0.5", "0 0.5 0.5 0.5 0.5 0.5", ""])
    def test_bad_token_count(self, line: str) -> None:
        """토큰 수가 5개가 아니면 ValueError."""
        with pytest.raises(ValueError, match="5개 값"):
            YoloLabel.from_line(line)

    def test_non_numeric_line(self) -> None:
        """숫자가 아닌 값이 섞이면 ValueError."""
        with pytest.raises(ValueError, match="숫자로 변환"):
            YoloLabel.from_line("0 a b c d")

    def test_invalid_image_size(self) -> None:
        """이미지 크기가 0 이면 ValueError."""
        with pytest.raises(ValueError, match="1px 이상"):
            YoloLabel(0, 0.5, 0.5, 0.5, 0.5).to_box_px(0, 100)
        with pytest.raises(ValueError, match="1px 이상"):
            YoloLabel.from_box_px(0, BoxPx(0, 0, 1, 1), 100, 0)

    def test_invalid_page_size(self) -> None:
        """페이지 크기가 잘못되면 ValueError."""
        with pytest.raises(ValueError, match="두 값"):
            YoloLabel(0, 0.5, 0.5, 0.5, 0.5).to_box_mm((210.0, 297.0, 1.0))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="0보다 커야"):
            YoloLabel(0, 0.5, 0.5, 0.5, 0.5).to_box_mm((0.0, 297.0))


# --------------------------------------------------------------------------
# 파일 입출력
# --------------------------------------------------------------------------


class TestFileIo:
    """라벨 파일 읽기·쓰기."""

    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        """파일 왕복이 무손실이다."""
        labels = [YoloLabel(0, 0.5, 0.7, 0.33, 0.03), YoloLabel(1, 0.15, 0.47, 0.03, 0.02)]
        path = write_yolo_labels(tmp_path / "nested" / "sample.txt", labels)
        assert read_yolo_labels(path) == labels

    def test_empty_file_is_valid(self, tmp_path: Path) -> None:
        """라벨이 없는 배경 전용 샘플도 빈 파일로 쓰고 읽을 수 있다."""
        path = write_yolo_labels(tmp_path / "empty.txt", [])
        assert path.read_text(encoding="utf-8") == ""
        assert read_yolo_labels(path) == []

    def test_comments_and_blank_lines_ignored(self, tmp_path: Path) -> None:
        """주석과 빈 줄은 건너뛴다."""
        path = tmp_path / "commented.txt"
        path.write_text(
            "# 주석\n\n0 0.5 0.5 0.2 0.1\n\n", encoding="utf-8"
        )
        assert read_yolo_labels(path) == [YoloLabel(0, 0.5, 0.5, 0.2, 0.1)]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """없는 파일은 조용히 빈 리스트를 주지 않고 FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="라벨 파일"):
            read_yolo_labels(tmp_path / "없음.txt")

    def test_bad_line_reports_line_number(self, tmp_path: Path) -> None:
        """형식이 틀린 줄은 줄 번호와 함께 ValueError."""
        path = tmp_path / "bad.txt"
        path.write_text("0 0.5 0.5 0.2 0.1\n1 0.5 0.5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="2번째 줄"):
            read_yolo_labels(path)

    def test_utf8_path_is_supported(self, tmp_path: Path) -> None:
        """한글 디렉터리·파일명에서도 동작한다(Windows 경로 규약)."""
        path = write_yolo_labels(
            tmp_path / "라벨" / "신청서_0001.txt", [YoloLabel(1, 0.5, 0.5, 0.03, 0.02)]
        )
        assert path.is_file()
        assert len(read_yolo_labels(path)) == 1


# --------------------------------------------------------------------------
# DocumentStructure → 라벨
# --------------------------------------------------------------------------


class TestStructureExport:
    """정답 구조에서 학습 라벨을 뽑는 경로."""

    def test_sample_structure_yields_expected_classes(
        self, sample_structure: DocumentStructure
    ) -> None:
        """서명란 2개(신청인·대리인) + 선택지 체크박스 2개가 라벨이 된다."""
        labels = structure_to_yolo_labels(sample_structure)
        counts = {name: 0 for name in CLASS_NAMES}
        for label in labels:
            counts[label.class_name] += 1
        assert counts == {"signature_field": 2, "checkbox": 2}

    def test_text_and_date_fields_are_not_labeled(
        self, sample_structure: DocumentStructure
    ) -> None:
        """TEXT_INPUT·DATE 항목은 탐지 대상이 아니므로 라벨에 들어가지 않는다."""
        labels = structure_to_yolo_labels(sample_structure)
        assert len(labels) == 4

    def test_coordinates_match_truth(self, sample_structure: DocumentStructure) -> None:
        """라벨을 mm 로 되돌리면 정답 상자와 0.01mm 이내로 일치한다."""
        labels = structure_to_yolo_labels(sample_structure)
        restored = [label.to_box_mm(sample_structure.page_size_mm) for label in labels]
        expected = [
            option.box_mm
            for item in sample_structure.fields
            for option in item.options
        ] + [
            item.box_mm
            for item in sample_structure.fields
            if item.type is FieldType.SIGNATURE and item.box_mm is not None
        ]
        for target in expected:
            assert any(
                box.x_mm == pytest.approx(target.x_mm, abs=0.01)
                and box.y_mm == pytest.approx(target.y_mm, abs=0.01)
                and box.w_mm == pytest.approx(target.w_mm, abs=0.01)
                and box.h_mm == pytest.approx(target.h_mm, abs=0.01)
                for box in restored
            ), f"정답 상자가 라벨에 없습니다: {target.to_tuple()}"

    def test_labels_sorted_top_to_bottom(self, sample_structure: DocumentStructure) -> None:
        """라벨은 위에서 아래 순서로 정렬되어 결정론적이다."""
        labels = structure_to_yolo_labels(sample_structure)
        centers = [label.cy for label in labels]
        assert centers == sorted(centers)
        assert structure_to_yolo_labels(sample_structure) == labels

    def test_export_to_file(
        self, sample_structure: DocumentStructure, tmp_path: Path
    ) -> None:
        """구조 → 파일 → 라벨 왕복이 무손실이다."""
        path = export_structure_labels(sample_structure, tmp_path / "truth.txt")
        assert read_yolo_labels(path) == structure_to_yolo_labels(sample_structure)

    def test_missing_box_is_warned_not_dropped_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """좌표 없는 서명란은 경고 로그를 남긴다(조용한 실패 금지)."""
        structure = DocumentStructure(
            document_id="no_box",
            fields=(
                Field(
                    id="signature_missing",
                    type=FieldType.SIGNATURE,
                    box_mm=None,
                    sensitivity=Sensitivity.PRIVATE,
                ),
            ),
        )
        with caplog.at_level("WARNING", logger="docagent.vision.dataset"):
            labels = structure_to_yolo_labels(structure)
        assert labels == []
        assert any("좌표가 없어" in record.message for record in caplog.records)

    def test_checkbox_field_is_labeled(self) -> None:
        """CHECKBOX 유형 항목 자체도 라벨이 된다(선택지가 없는 단독 체크박스)."""
        structure = DocumentStructure(
            document_id="single_checkbox",
            fields=(
                Field(
                    id="agree",
                    type=FieldType.CHECKBOX,
                    box_mm=BoxMm(26.0, 139.0, 6.0, 6.0),
                ),
            ),
        )
        labels = structure_to_yolo_labels(structure)
        assert [label.class_name for label in labels] == ["checkbox"]

    def test_option_boxes_of_choice_field(self) -> None:
        """CHOICE 항목의 선택지 네모 칸이 각각 체크박스 라벨이 된다."""
        structure = DocumentStructure(
            document_id="choice",
            fields=(
                Field(
                    id="consent",
                    type=FieldType.CHOICE,
                    options=(
                        Option("동의함", BoxMm(26.0, 139.0, 6.0, 6.0)),
                        Option("동의하지 않음", BoxMm(86.0, 139.0, 6.0, 6.0)),
                    ),
                ),
            ),
        )
        assert len(structure_to_yolo_labels(structure)) == 2

    def test_checkbox_field_with_one_option_yields_a_single_label(self) -> None:
        """체크박스 1개짜리 항목은 라벨을 1건만 만든다.

        구조화는 체크박스 항목의 ``box_mm`` 을 제목 행까지 포함한 합집합으로 잡고
        같은 칸을 ``options[0]`` 로도 담는다. 둘 다 내보내면 같은 칸이 두 번
        라벨되고 그중 하나는 문구 블록 전체라서 학습셋이 오염된다.
        """
        option_box = BoxMm(30.0, 121.0, 5.0, 5.0)
        structure = DocumentStructure(
            document_id="checkbox_with_option",
            fields=(
                Field(
                    id="checkbox_01",
                    type=FieldType.CHECKBOX,
                    options=(Option("동의함", option_box),),
                    # 제목 행까지 포함한 합집합 상자(체크박스가 아니다).
                    box_mm=BoxMm(25.0, 110.0, 39.0, 16.0),
                ),
            ),
        )
        labels = structure_to_yolo_labels(structure)
        assert len(labels) == 1
        assert labels[0].class_name == "checkbox"
        restored = labels[0].to_box_mm(structure.page_size_mm)
        assert restored.w_mm == pytest.approx(option_box.w_mm, abs=0.05)
        assert restored.h_mm == pytest.approx(option_box.h_mm, abs=0.05)


# --------------------------------------------------------------------------
# COCO → YOLO
# --------------------------------------------------------------------------


def _coco_payload() -> dict:
    """왕복 검증용 최소 COCO 문서를 만든다.

    :returns: ``images`` / ``annotations`` / ``categories`` 를 갖춘 dict.
    """
    return {
        "images": [
            {"id": 1, "file_name": "form_0001.png", "width": 1000, "height": 2000},
            {"id": 2, "file_name": "form_0002.png", "width": 1000, "height": 2000},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [100, 200, 300, 40]},
            {"id": 2, "image_id": 1, "category_id": 20, "bbox": [500, 900, 40, 40]},
        ],
        "categories": [
            {"id": 10, "name": "signature_field"},
            {"id": 20, "name": "checkbox"},
        ],
    }


class TestCocoConversion:
    """COCO JSON → YOLO txt 변환."""

    def test_round_trip_boxes(self, tmp_path: Path) -> None:
        """COCO bbox → YOLO → 픽셀 상자 왕복 오차가 1px 이내다."""
        written = coco_to_yolo(_coco_payload(), tmp_path)
        labels = read_yolo_labels(written["form_0001.png"])
        boxes = sorted(
            (label.class_id, label.to_box_px(1000, 2000).to_tuple()) for label in labels
        )
        assert boxes[0][0] == 0
        for got, want in zip(boxes[0][1], (100, 200, 300, 40)):
            assert abs(got - want) <= 1
        assert boxes[1][0] == 1
        for got, want in zip(boxes[1][1], (500, 900, 40, 40)):
            assert abs(got - want) <= 1

    def test_image_without_annotation_gets_empty_file(self, tmp_path: Path) -> None:
        """어노테이션이 없는 이미지도 빈 라벨 파일을 만든다(배경 샘플 보존)."""
        written = coco_to_yolo(_coco_payload(), tmp_path)
        assert written["form_0002.png"].is_file()
        assert read_yolo_labels(written["form_0002.png"]) == []

    def test_reads_json_file(self, tmp_path: Path) -> None:
        """파일 경로로도 읽을 수 있다."""
        source = tmp_path / "coco.json"
        source.write_text(
            json.dumps(_coco_payload(), ensure_ascii=False), encoding="utf-8"
        )
        written = coco_to_yolo(source, tmp_path / "labels")
        assert len(written) == 2

    def test_unknown_category_raises(self, tmp_path: Path) -> None:
        """매핑에 없는 카테고리는 조용히 무시하지 않고 ValueError."""
        payload = _coco_payload()
        payload["categories"][0]["name"] = "미정의_클래스"
        with pytest.raises(ValueError, match="매핑되지 않은 카테고리"):
            coco_to_yolo(payload, tmp_path)

    def test_unknown_category_can_be_skipped(self, tmp_path: Path) -> None:
        """``skip_unknown=True`` 면 경고 후 건너뛴다."""
        payload = _coco_payload()
        payload["categories"][0]["name"] = "미정의_클래스"
        written = coco_to_yolo(payload, tmp_path, skip_unknown=True)
        assert len(read_yolo_labels(written["form_0001.png"])) == 1

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        """필수 키가 없으면 ValueError."""
        with pytest.raises(ValueError, match="'annotations'"):
            coco_to_yolo({"images": [], "categories": []}, tmp_path)

    def test_dangling_image_id_raises(self, tmp_path: Path) -> None:
        """존재하지 않는 image_id 를 가리키면 ValueError."""
        payload = _coco_payload()
        payload["annotations"][0]["image_id"] = 999
        with pytest.raises(ValueError, match="image_id"):
            coco_to_yolo(payload, tmp_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """없는 JSON 경로는 FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="COCO JSON"):
            coco_to_yolo(tmp_path / "없음.json", tmp_path)


# --------------------------------------------------------------------------
# 분할
# --------------------------------------------------------------------------


class TestSplitDataset:
    """결정론적 데이터셋 분할."""

    def test_counts_follow_ratios(self) -> None:
        """250건을 8:1:1 로 나누면 200/25/25 가 된다(로드맵 계획치)."""
        files = [f"form_{index:04d}.png" for index in range(250)]
        split = split_dataset(files, DEFAULT_SPLIT_RATIOS, seed=7)
        assert split.counts == {"train": 200, "val": 25, "test": 25}
        assert split.total == 250

    def test_deterministic_for_same_seed(self) -> None:
        """같은 시드는 항상 같은 결과를 낸다."""
        files = [f"f{index}.png" for index in range(37)]
        assert split_dataset(files, seed=11) == split_dataset(files, seed=11)

    def test_independent_of_input_order(self) -> None:
        """입력 순서가 달라도 결과가 같다(내부에서 먼저 정렬한다)."""
        files = [f"f{index}.png" for index in range(37)]
        assert split_dataset(files, seed=11) == split_dataset(
            list(reversed(files)), seed=11
        )

    def test_different_seed_changes_split(self) -> None:
        """시드가 다르면 배치가 달라진다."""
        files = [f"f{index}.png" for index in range(100)]
        assert split_dataset(files, seed=1) != split_dataset(files, seed=2)

    def test_splits_are_disjoint_and_complete(self) -> None:
        """분할 사이에 중복이 없고 전체를 빠짐없이 덮는다."""
        files = [f"f{index}.png" for index in range(53)]
        split = split_dataset(files, seed=3)
        union = set(split.train) | set(split.val) | set(split.test)
        assert union == set(files)
        assert split.total == len(files)

    def test_empty_input(self) -> None:
        """빈 입력은 빈 분할을 돌려준다."""
        assert split_dataset([]) == DatasetSplit()

    def test_accepts_paths(self, tmp_path: Path) -> None:
        """Path 객체 목록도 받는다(문자열로 정규화)."""
        split = split_dataset([tmp_path / "a.png", tmp_path / "b.png"], (1.0, 0.0, 0.0))
        assert split.counts == {"train": 2, "val": 0, "test": 0}

    def test_to_dict(self) -> None:
        """JSON 직렬화 가능한 dict 를 돌려준다."""
        payload = split_dataset(["a", "b"], (1.0, 0.0, 0.0)).to_dict()
        assert json.loads(json.dumps(payload)) == payload

    @pytest.mark.parametrize(
        ("ratios", "message"),
        [
            ((0.5, 0.5), "세 값"),
            ((0.5, 0.5, 0.5), "합은 1.0"),
            ((-0.1, 0.6, 0.5), "음수"),
        ],
    )
    def test_invalid_ratios(self, ratios: tuple, message: str) -> None:
        """잘못된 비율은 한국어 ValueError."""
        with pytest.raises(ValueError, match=message):
            split_dataset(["a", "b"], ratios)

    def test_duplicate_files_rejected(self) -> None:
        """중복 파일은 데이터 누수를 만들므로 거부한다."""
        with pytest.raises(ValueError, match="중복"):
            split_dataset(["a.png", "a.png", "b.png"])


# --------------------------------------------------------------------------
# data.yaml
# --------------------------------------------------------------------------


class TestDataYaml:
    """ultralytics 학습 정의 파일 생성."""

    def test_contains_required_keys(self, tmp_path: Path) -> None:
        """path/train/val/test/nc/names 를 모두 적는다."""
        path = write_data_yaml(tmp_path / "data.yaml", tmp_path / "dataset")
        text = path.read_text(encoding="utf-8")
        for key in ("path:", "train:", "val:", "test:", "nc: 2", "names:"):
            assert key in text
        assert "0: signature_field" in text
        assert "1: checkbox" in text

    def test_uses_posix_separators(self, tmp_path: Path) -> None:
        """경로는 POSIX 슬래시로 적는다(Windows 역슬래시 금지)."""
        path = write_data_yaml(tmp_path / "data.yaml", tmp_path / "a" / "b")
        line = next(
            row for row in path.read_text(encoding="utf-8").splitlines() if row.startswith("path:")
        )
        assert "\\" not in line
        assert "a/b" in line

    def test_test_split_optional(self, tmp_path: Path) -> None:
        """``test=None`` 이면 test 항목을 적지 않는다."""
        path = write_data_yaml(tmp_path / "data.yaml", tmp_path, test=None)
        assert "test:" not in path.read_text(encoding="utf-8")

    def test_rejects_empty_or_duplicated_names(self, tmp_path: Path) -> None:
        """클래스 이름이 비었거나 중복이면 ValueError."""
        with pytest.raises(ValueError, match="비어 있습니다"):
            write_data_yaml(tmp_path / "a.yaml", tmp_path, class_names=[])
        with pytest.raises(ValueError, match="중복"):
            write_data_yaml(tmp_path / "b.yaml", tmp_path, class_names=["a", "a"])


# --------------------------------------------------------------------------
# 검증
# --------------------------------------------------------------------------


class TestValidation:
    """라벨 검증기 — 문제를 예외 대신 보고서로 모은다."""

    def test_valid_labels_report_ok(self) -> None:
        """정상 라벨은 문제 0건."""
        report = validate_labels(["0 0.5 0.5 0.2 0.1", "1 0.2 0.3 0.05 0.05"])
        assert report.ok
        assert report.label_count == 2
        assert "정상" in report.format_report()

    def test_out_of_range_coordinate(self) -> None:
        """0~1 범위를 벗어난 좌표를 잡아낸다."""
        report = validate_labels(["0 1.5 0.5 0.2 0.1"])
        assert not report.ok
        assert any("cx" in issue for issue in report.issues)

    def test_non_positive_size(self) -> None:
        """폭·높이가 0 이하이면 문제로 본다."""
        report = validate_labels(["0 0.5 0.5 0.0 0.1"])
        assert any("w=" in issue for issue in report.issues)

    def test_box_outside_image(self) -> None:
        """중심 ± 절반이 이미지를 벗어나면 경계 이탈로 보고한다."""
        report = validate_labels(["0 0.95 0.5 0.2 0.1"])
        assert any("좌우 경계" in issue for issue in report.issues)
        report_y = validate_labels(["0 0.5 0.98 0.2 0.1"])
        assert any("상하 경계" in issue for issue in report_y.issues)

    def test_unknown_class_id(self) -> None:
        """정의되지 않은 클래스 id 를 잡아낸다."""
        report = validate_labels(["9 0.5 0.5 0.2 0.1"])
        assert any("정의되지 않은 클래스" in issue for issue in report.issues)

    def test_wrong_token_count(self) -> None:
        """토큰 수가 틀린 줄을 줄 번호와 함께 보고한다."""
        report = validate_labels(["0 0.5 0.5 0.2"])
        assert any("1번째 줄" in issue for issue in report.issues)

    def test_empty_label_set(self) -> None:
        """라벨이 하나도 없으면 기본적으로 문제로 본다."""
        assert not validate_labels([]).ok
        assert validate_labels([], allow_empty=True).ok

    def test_report_formatting_lists_issues(self) -> None:
        """문제 보고서는 상세 목록을 함께 담는다."""
        text = validate_labels(["9 0.5 0.5 0.2 0.1"]).format_report()
        assert "문제" in text
        assert "  - " in text

    def test_report_to_dict_is_json_safe(self) -> None:
        """보고서는 JSON 직렬화 가능하다."""
        payload = validate_labels(["0 0.5 0.5 0.2 0.1"]).to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_report_rejects_negative_count(self) -> None:
        """음수 라벨 개수는 ValueError."""
        with pytest.raises(ValueError, match="0 이상"):
            LabelReport(source="x", label_count=-1)

    def test_validate_file(self, tmp_path: Path) -> None:
        """파일 단위 검증이 동작한다."""
        path = write_yolo_labels(tmp_path / "ok.txt", [YoloLabel(0, 0.5, 0.5, 0.2, 0.1)])
        assert validate_label_file(path).ok

    def test_validate_missing_file_is_reported_not_raised(self, tmp_path: Path) -> None:
        """없는 파일은 예외가 아니라 보고서의 문제로 남는다."""
        report = validate_label_file(tmp_path / "없음.txt")
        assert not report.ok
        assert any("찾을 수 없습니다" in issue for issue in report.issues)

    def test_validate_dir(self, tmp_path: Path) -> None:
        """디렉터리 전체를 검증하고 파일명 순으로 보고한다."""
        write_yolo_labels(tmp_path / "a.txt", [YoloLabel(0, 0.5, 0.5, 0.2, 0.1)])
        (tmp_path / "b.txt").write_text("9 0.5 0.5 0.2 0.1\n", encoding="utf-8")
        reports = validate_label_dir(tmp_path)
        assert [Path(item.source).name for item in reports] == ["a.txt", "b.txt"]
        assert reports[0].ok and not reports[1].ok

    def test_validate_dir_missing(self, tmp_path: Path) -> None:
        """없는 디렉터리는 FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="라벨 디렉터리"):
            validate_label_dir(tmp_path / "없음")

    def test_exported_truth_passes_validation(
        self, sample_structure: DocumentStructure, tmp_path: Path
    ) -> None:
        """합성 정답에서 뽑은 라벨은 검증을 통과한다(파이프라인 일관성)."""
        path = export_structure_labels(sample_structure, tmp_path / "truth.txt")
        report = validate_label_file(path)
        assert report.ok, report.format_report()
        assert report.label_count == 4
