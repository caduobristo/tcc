import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import numpy as np

from scripts.audit_datasets import checked_write
from src.data.archive import GuitarFxArchive
from src.data.paths import baseline_scenario_root


class DataCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "archives/guitar_fx_dist"
        self.folder.mkdir(parents=True)

    def make_archive(self, array=None):
        if array is None:
            array = np.ones((198, 128), dtype=np.float32)
        stream = io.BytesIO()
        np.save(stream, array)
        path = self.folder / "mono_cont_mel16-001.zip"
        with ZipFile(path, "w", compression=ZIP_STORED) as archive:
            archive.writestr("Mono_Continuous/808/sample.npy", stream.getvalue())
        return path

    def test_reads_array_without_extracting_and_preserves_archive(self):
        path = self.make_archive()
        original = path.read_bytes()
        with GuitarFxArchive("mono_cont", "mel16", self.root) as archive:
            self.assertEqual(archive.class_counts(), {"808": 1})
            np.testing.assert_array_equal(archive.read_feature("808", "sample"), np.ones((198, 128)))
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "processed").exists())

    def test_ambiguous_archive_selection_fails(self):
        self.make_archive()
        (self.folder / "mono_cont_mel16-002.zip").write_bytes(b"different")
        with self.assertRaises(FileNotFoundError):
            GuitarFxArchive("mono_cont", "mel16", self.root)

    def test_corrupt_payload_is_detected_by_crc(self):
        path = self.make_archive()
        contents = bytearray(path.read_bytes())
        contents[contents.index(b"\x93NUMPY") + 200] ^= 1
        path.write_bytes(contents)
        with GuitarFxArchive("mono_cont", "mel16", self.root) as archive:
            with self.assertRaises(BadZipFile):
                archive.read_feature("808", "sample")

    def test_wrong_shape_and_nonfinite_values_are_rejected(self):
        for array in (np.ones((87, 128), dtype=np.float32),
                      np.full((198, 128), np.nan, dtype=np.float32)):
            with self.subTest(shape=array.shape):
                self.make_archive(array)
                with GuitarFxArchive("mono_cont", "mel16", self.root) as archive:
                    with self.assertRaises(ValueError):
                        archive.read_feature("808", "sample")

    def test_pickle_arrays_are_not_loaded(self):
        self.make_archive(np.array([{"unsafe": "object"}], dtype=object))
        with GuitarFxArchive("mono_cont", "mel16", self.root) as archive:
            with self.assertRaises(ValueError):
                archive.read_feature("808", "sample")

    def test_path_components_cannot_escape_archive(self):
        self.make_archive()
        with GuitarFxArchive("mono_cont", "mel16", self.root) as archive:
            with self.assertRaises(ValueError):
                archive.read_feature("../808", "sample")

    def test_baseline_never_falls_back_to_another_representation(self):
        (self.root / "processed/guitar_fx_dist/mel32/Mono_Continuous/Features").mkdir(parents=True)
        with patch.dict(os.environ, {"TCC_DATA_ROOT": str(self.root)}):
            with self.assertRaisesRegex(FileNotFoundError, "not replacements"):
                baseline_scenario_root("mono_cont")

    def test_existing_different_data_is_not_overwritten(self):
        path = self.root / "metadata/protected.csv"
        checked_write(path, b"original")
        checked_write(path, b"original")
        with self.assertRaises(FileExistsError):
            checked_write(path, b"changed")
        self.assertEqual(path.read_bytes(), b"original")

    def test_parameter_decimals_in_sample_names_are_preserved(self):
        self.make_archive()
        sample = "G63-49109-808-D2.7-20680"
        metadata = self.root / "metadata/guitar_fx_dist/validated/Mono_Continuous/808/proc_settings.csv"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"filename,fx,gain\n{sample}.wav,808,0.27\n", encoding="utf-8")
        with GuitarFxArchive("mono_cont", "mel16", self.root) as archive:
            self.assertEqual(archive.read_settings("808", sample)["gain"], "0.27")
            self.assertEqual(archive.read_settings("808", sample+".npy")["filename"], sample+".wav")


if __name__ == "__main__":
    unittest.main()
