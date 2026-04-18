from pathlib import Path

import pytest
pd = pytest.importorskip("pandas")
pytest.importorskip("PIL.Image")
from PIL import Image

from brain_mri.ml import dataset_builder as dataset_builder_module
from brain_mri.ml.dataset_builder import (
    DatasetBuilderConfig,
    _coerce_config,
    _display_path,
    _normalize_base_dirs,
    _parse_mri_subject_ids,
    create_exam_level_dataset,
    list_orientation_paths,
    populate_orientation_paths,
)


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (16, 16), color=int(value))
    image.save(path)


def _write_mock_exam(dataset_root: Path, mri_id: str, value: int = 64) -> None:
    for orient in ["axl", "cor", "sag"]:
        _write_png(dataset_root / orient / f"{mri_id}_{orient}.png", value)


def test_list_orientation_paths_returns_available_relative_paths(tmp_path):
    dataset_root = tmp_path / "dataset"
    mri_id = "OAS2_0001_MR1"
    _write_mock_exam(dataset_root, mri_id, value=96)

    paths = list_orientation_paths(mri_id, dataset_root)

    assert paths == [
        "axl/OAS2_0001_MR1_axl.png",
        "cor/OAS2_0001_MR1_cor.png",
        "sag/OAS2_0001_MR1_sag.png",
    ]


def test_populate_orientation_paths_enriches_dataframe_rows(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_mock_exam(dataset_root, "OAS2_0001_MR1", value=32)
    _write_png(dataset_root / "axl" / "OAS2_0002_MR1_axl.png", value=64)

    df = pd.DataFrame(
        [
            {"MRI_ID": "OAS2_0001_MR1", "original_path": ""},
            {"MRI_ID": "OAS2_0002_MR1", "original_path": ""},
        ]
    )

    enriched = populate_orientation_paths(df, dataset_root)

    assert list(enriched["MRI_ID"]) == ["OAS2_0001_MR1", "OAS2_0002_MR1"]
    assert enriched.iloc[0]["orientation_paths"] == [
        "axl/OAS2_0001_MR1_axl.png",
        "cor/OAS2_0001_MR1_cor.png",
        "sag/OAS2_0001_MR1_sag.png",
    ]
    assert enriched.iloc[0]["original_path"] == "axl/OAS2_0001_MR1_axl.png"
    assert enriched.iloc[1]["orientation_paths"] == ["axl/OAS2_0002_MR1_axl.png"]
    assert enriched.iloc[1]["original_path"] == "axl/OAS2_0002_MR1_axl.png"


@pytest.mark.skipif(not dataset_builder_module.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_create_exam_level_dataset_writes_split_csv_with_subject_partitions(tmp_path):
    dataset_root = tmp_path
    output_dir = tmp_path / "output"
    csv_path = tmp_path / "oasis_longitudinal_demographic.csv"
    descriptors_csv = output_dir / "ventricle_descriptors.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    demo_rows = []
    descriptor_rows = []
    for index in range(10):
        subject_id = f"OAS2_{index + 1:04d}"
        mri_id = f"{subject_id}_MR1"
        group = "Demented" if index % 2 else "Nondemented"
        sex = "F" if index % 2 else "M"
        _write_mock_exam(dataset_root, mri_id, value=40 + index)
        demo_rows.append(
            {
                "MRI ID": mri_id,
                "Subject ID": subject_id,
                "Group": group,
                "M/F": sex,
                "Age": 60 + index,
                "EDUC": 12 + (index % 3),
                "MMSE": 28 - (index % 4),
                "CDR": 1.0 if group == "Demented" else 0.0,
                "eTIV": 1500 + index,
                "nWBV": 0.70 + index * 0.01,
                "ASF": 1.10 + index * 0.01,
            }
        )
        descriptor_rows.append(
            {
                "MRI_ID": mri_id,
                "viable": True,
                "ventricle_area": 100.0 + index,
                "ventricle_perimeter": 50.0 + index,
                "ventricle_circularity": 0.80 - index * 0.01,
                "ventricle_eccentricity": 0.20 + index * 0.01,
            }
        )

    pd.DataFrame(demo_rows).to_csv(csv_path, sep=";", decimal=",", index=False)
    pd.DataFrame(descriptor_rows).to_csv(descriptors_csv, index=False)

    config = DatasetBuilderConfig(
        dataset_dir=dataset_root,
        output_dir=output_dir,
        csv_path=csv_path,
        descriptors_csv=descriptors_csv,
    )

    dataset_df, output_path = create_exam_level_dataset(config)

    assert output_path.exists()
    assert len(dataset_df) == 10
    assert set(dataset_df["split"]) == {"train", "validation", "test"}
    assert set(dataset_df["Final_Group"]) == {"Demented", "Nondemented"}

    subjects_by_split = {
        split: set(dataset_df.loc[dataset_df["split"] == split, "Subject_ID"])
        for split in ["train", "validation", "test"]
    }
    assert subjects_by_split["train"].isdisjoint(subjects_by_split["validation"])
    assert subjects_by_split["train"].isdisjoint(subjects_by_split["test"])
    assert subjects_by_split["validation"].isdisjoint(subjects_by_split["test"])

    for split_name in ["train", "validation", "test"]:
        split_df = dataset_df[dataset_df["split"] == split_name]
        assert not split_df.empty
        split_set = set(split_df["Final_Group"])
        if split_name == "test":
            assert split_set.issubset({"Demented", "Nondemented"})
            assert len(split_set) >= 1
        else:
            assert split_set == {"Demented", "Nondemented"}

    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == len(dataset_df)


# ---------------------------------------------------------------------------
# Additional edge-case and unit tests
# ---------------------------------------------------------------------------


# --- _normalize_base_dirs ---------------------------------------------------


def test_normalize_base_dirs_string(tmp_path):
    """A plain string must be normalised to a one-element list of Path."""
    result = _normalize_base_dirs(str(tmp_path))
    assert result == [tmp_path]


def test_normalize_base_dirs_path(tmp_path):
    """A Path object must be normalised to a one-element list."""
    result = _normalize_base_dirs(tmp_path)
    assert result == [tmp_path]


def test_normalize_base_dirs_iterable(tmp_path):
    """An iterable of strings must be converted to a list of Paths."""
    dirs = [str(tmp_path), str(tmp_path / "sub")]
    result = _normalize_base_dirs(dirs)
    assert result == [tmp_path, tmp_path / "sub"]


def test_normalize_base_dirs_invalid_type():
    """Passing an integer must raise TypeError."""
    with pytest.raises(TypeError):
        _normalize_base_dirs(42)


# --- _parse_mri_subject_ids -------------------------------------------------


def test_parse_mri_subject_ids_normal():
    """Standard MRI filename parses to correct MRI_ID and Subject_ID."""
    mri_id, subj_id = _parse_mri_subject_ids("OAS2_0001_MR1_axl.png")
    assert mri_id == "OAS2_0001_MR1"
    assert subj_id == "OAS2_0001"


def test_parse_mri_subject_ids_nifti_ext():
    """NIfTI .nii.gz extension is stripped before parsing."""
    mri_id, subj_id = _parse_mri_subject_ids("OAS2_0005_MR2_axl.nii.gz")
    assert mri_id == "OAS2_0005_MR2"
    assert subj_id == "OAS2_0005"


def test_parse_mri_subject_ids_too_short():
    """Filenames with fewer than 3 underscore-separated parts return (None, None)."""
    mri_id, subj_id = _parse_mri_subject_ids("short.png")
    assert mri_id is None
    assert subj_id is None


def test_parse_mri_subject_ids_exactly_three_parts():
    """Filenames with exactly 3 underscore parts must work correctly."""
    mri_id, subj_id = _parse_mri_subject_ids("OAS_001_MR1.png")
    assert mri_id == "OAS_001_MR1"
    assert subj_id == "OAS_001"


# --- _coerce_config ---------------------------------------------------------


def test_coerce_config_from_dataclass(tmp_path):
    """_coerce_config must round-trip a DatasetBuilderConfig unchanged."""
    cfg = DatasetBuilderConfig(
        dataset_dir=tmp_path / "axl",
        output_dir=tmp_path / "out",
        csv_path=tmp_path / "demo.csv",
        descriptors_csv=tmp_path / "desc.csv",
    )
    coerced = _coerce_config(cfg)
    assert coerced.dataset_dir == Path(tmp_path / "axl")
    assert coerced.output_dir == Path(tmp_path / "out")


def test_coerce_config_from_dict(tmp_path):
    """_coerce_config must accept a plain dict mapping."""
    mapping = {
        "dataset_dir": str(tmp_path / "axl"),
        "output_dir": str(tmp_path / "out"),
        "csv_path": str(tmp_path / "demo.csv"),
        "descriptors_csv": str(tmp_path / "desc.csv"),
    }
    coerced = _coerce_config(mapping)
    assert isinstance(coerced, DatasetBuilderConfig)
    assert coerced.dataset_dir == tmp_path / "axl"


def test_coerce_config_invalid_type():
    """_coerce_config must raise TypeError for unsupported config types."""
    with pytest.raises(TypeError):
        _coerce_config(["not", "a", "config"])


# --- list_orientation_paths -------------------------------------------------


def test_list_orientation_paths_empty_when_no_files(tmp_path):
    """Returns an empty list if none of the expected image files exist."""
    paths = list_orientation_paths("OAS2_9999_MR1", tmp_path)
    assert paths == []


def test_list_orientation_paths_partial_orientations(tmp_path):
    """Only available orientations appear in the result."""
    dataset_root = tmp_path / "dataset"
    # Create only axial image
    axl_path = dataset_root / "axl" / "OAS2_0010_MR1_axl.png"
    axl_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=0).save(axl_path)

    paths = list_orientation_paths("OAS2_0010_MR1", dataset_root)

    assert len(paths) == 1
    assert "axl" in paths[0]


def test_list_orientation_paths_multiple_base_dirs(tmp_path):
    """Paths from multiple base directories are combined without duplicates."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    for root in (root1, root2):
        img = root / "axl" / "OAS2_0011_MR1_axl.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 8), color=0).save(img)

    paths = list_orientation_paths("OAS2_0011_MR1", [root1, root2])
    assert len(paths) == 2


# --- populate_orientation_paths ---------------------------------------------


def test_populate_orientation_paths_skips_none_mri_id(tmp_path):
    """Rows where MRI_ID is not a string must be silently skipped."""
    df = pd.DataFrame([{"MRI_ID": None, "original_path": ""}])
    result = populate_orientation_paths(df, tmp_path)
    assert len(result) == 0


def test_populate_orientation_paths_empty_dataframe(tmp_path):
    """An empty dataframe must be returned unchanged (None-safe)."""
    df = pd.DataFrame(columns=["MRI_ID", "original_path"])
    result = populate_orientation_paths(df, tmp_path)
    assert result is not None
    assert len(result) == 0


def test_populate_orientation_paths_keeps_row_with_original_path_when_no_images(tmp_path):
    """Rows with no orientation files but a pre-existing original_path are retained."""
    df = pd.DataFrame([{"MRI_ID": "OAS2_9998_MR1", "original_path": "axl/OAS2_9998_MR1_axl.png"}])
    result = populate_orientation_paths(df, tmp_path)
    # Row should be kept because original_path is non-empty
    assert len(result) == 1
    assert result.iloc[0]["original_path"] == "axl/OAS2_9998_MR1_axl.png"


# --- create_exam_level_dataset additional cases ----------------------------


@pytest.mark.skipif(not dataset_builder_module.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_create_exam_level_dataset_without_descriptors_csv(tmp_path):
    """Dataset creation must succeed even when the descriptors CSV does not exist."""
    dataset_root = tmp_path
    output_dir = tmp_path / "output"
    csv_path = tmp_path / "demo.csv"
    descriptors_csv = output_dir / "no_descriptors.csv"  # intentionally absent

    output_dir.mkdir(parents=True, exist_ok=True)

    demo_rows = []
    for index in range(10):
        subject_id = f"OAS2_{index + 1:04d}"
        mri_id = f"{subject_id}_MR1"
        group = "Demented" if index % 2 else "Nondemented"
        _write_mock_exam(dataset_root, mri_id, value=50 + index)
        demo_rows.append(
            {
                "MRI ID": mri_id,
                "Subject ID": subject_id,
                "Group": group,
                "M/F": "M",
                "Age": 65 + index,
                "EDUC": 12,
                "MMSE": 28,
                "CDR": 0.5 if group == "Demented" else 0.0,
                "eTIV": 1500,
                "nWBV": 0.70,
                "ASF": 1.10,
            }
        )

    pd.DataFrame(demo_rows).to_csv(csv_path, sep=";", decimal=",", index=False)

    config = DatasetBuilderConfig(
        dataset_dir=dataset_root,
        output_dir=output_dir,
        csv_path=csv_path,
        descriptors_csv=descriptors_csv,
    )

    dataset_df, output_path = create_exam_level_dataset(config)

    assert output_path.exists()
    assert len(dataset_df) == 10
    # has_descriptors should be False for all rows (no descriptor file)
    assert not dataset_df["has_descriptors"].any()


@pytest.mark.skipif(not dataset_builder_module.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_create_exam_level_dataset_converted_group_resolution(tmp_path):
    """'Converted' subjects with CDR > 0 must become 'Demented' in Final_Group."""
    dataset_root = tmp_path
    output_dir = tmp_path / "output"
    csv_path = tmp_path / "demo.csv"
    descriptors_csv = output_dir / "no_desc.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    demo_rows = []
    for index in range(10):
        subject_id = f"OAS2_{index + 1:04d}"
        mri_id = f"{subject_id}_MR1"
        # Mix of group types: Demented, Nondemented, Converted-with-CDR
        if index < 4:
            group = "Demented"
            cdr = 1.0
        elif index < 7:
            group = "Nondemented"
            cdr = 0.0
        else:
            group = "Converted"
            cdr = 0.5  # CDR > 0 → should become Demented
        _write_mock_exam(dataset_root, mri_id, value=50 + index)
        demo_rows.append(
            {
                "MRI ID": mri_id,
                "Subject ID": subject_id,
                "Group": group,
                "M/F": "M",
                "Age": 65 + index,
                "EDUC": 12,
                "MMSE": 28,
                "CDR": cdr,
                "eTIV": 1500,
                "nWBV": 0.70,
                "ASF": 1.10,
            }
        )

    pd.DataFrame(demo_rows).to_csv(csv_path, sep=";", decimal=",", index=False)

    config = DatasetBuilderConfig(
        dataset_dir=dataset_root,
        output_dir=output_dir,
        csv_path=csv_path,
        descriptors_csv=descriptors_csv,
    )

    dataset_df, _ = create_exam_level_dataset(config)

    # All converted+CDR>0 rows must have Final_Group == 'Demented'
    assert "Converted" not in dataset_df["Final_Group"].values


@pytest.mark.skipif(not dataset_builder_module.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_create_exam_level_dataset_insufficient_subjects_raises(tmp_path):
    """Fewer than 3 subjects must raise ValueError."""
    dataset_root = tmp_path
    output_dir = tmp_path / "output"
    csv_path = tmp_path / "demo.csv"
    descriptors_csv = output_dir / "no_desc.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    demo_rows = []
    for index in range(2):  # Only 2 subjects
        subject_id = f"OAS2_{index + 1:04d}"
        mri_id = f"{subject_id}_MR1"
        group = "Demented" if index % 2 else "Nondemented"
        _write_mock_exam(dataset_root, mri_id, value=50)
        demo_rows.append(
            {
                "MRI ID": mri_id,
                "Subject ID": subject_id,
                "Group": group,
                "M/F": "M",
                "Age": 65,
                "EDUC": 12,
                "MMSE": 28,
                "CDR": 0.0,
                "eTIV": 1500,
                "nWBV": 0.70,
                "ASF": 1.10,
            }
        )

    pd.DataFrame(demo_rows).to_csv(csv_path, sep=";", decimal=",", index=False)

    config = DatasetBuilderConfig(
        dataset_dir=dataset_root,
        output_dir=output_dir,
        csv_path=csv_path,
        descriptors_csv=descriptors_csv,
    )

    with pytest.raises(ValueError, match="insuficientes|mínimo"):
        create_exam_level_dataset(config)


def test_dataset_builder_config_is_frozen(tmp_path):
    """DatasetBuilderConfig is a frozen dataclass and must not allow mutation."""
    cfg = DatasetBuilderConfig(
        dataset_dir=tmp_path,
        output_dir=tmp_path,
        csv_path=tmp_path / "demo.csv",
        descriptors_csv=tmp_path / "desc.csv",
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.dataset_dir = tmp_path / "other"

# ---------------------------------------------------------------------------
# Tests for _display_path (new helper added in this PR)
# ---------------------------------------------------------------------------


def test_display_path_within_base_dir_returns_posix_relative(tmp_path):
    base_dir = tmp_path / "dataset"
    base_dir.mkdir()
    image_path = base_dir / "axl" / "OAS2_0001_MR1_axl.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")

    result = _display_path(image_path, base_dir)

    assert result == "axl/OAS2_0001_MR1_axl.png"


def test_display_path_outside_base_dir_returns_absolute_posix(tmp_path):
    base_dir = tmp_path / "dataset"
    base_dir.mkdir()
    outside_image = tmp_path / "other" / "axl" / "OAS2_0001_MR1_axl.png"
    outside_image.parent.mkdir(parents=True)
    outside_image.write_bytes(b"")

    result = _display_path(outside_image, base_dir)

    assert result == outside_image.as_posix()
    assert Path(result).is_absolute()


def test_display_path_nested_subdirectory_uses_forward_slashes(tmp_path):
    base_dir = tmp_path / "root"
    base_dir.mkdir()
    image_path = base_dir / "a" / "b" / "c" / "file.nii.gz"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")

    result = _display_path(image_path, base_dir)

    assert result == "a/b/c/file.nii.gz"
    assert "\\" not in result


# ---------------------------------------------------------------------------
# Tests for list_orientation_paths deduplication (new seen_files logic)
# ---------------------------------------------------------------------------


def test_list_orientation_paths_deduplicates_same_file_across_base_dirs(tmp_path):
    """Passing the same base_dir twice must not return duplicate paths."""
    dataset_root = tmp_path / "dataset"
    mri_id = "OAS2_0001_MR1"
    _write_mock_exam(dataset_root, mri_id, value=128)

    paths = list_orientation_paths(mri_id, [dataset_root, dataset_root])

    assert paths.count("axl/OAS2_0001_MR1_axl.png") == 1
    assert paths.count("cor/OAS2_0001_MR1_cor.png") == 1
    assert paths.count("sag/OAS2_0001_MR1_sag.png") == 1
    assert len(paths) == 3
