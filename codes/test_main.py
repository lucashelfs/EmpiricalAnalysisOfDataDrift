from unittest import TestCase

# class TryTesting(TestCase):
#     def test_always_passes(self):
#         self.assertTrue(True)
#
#     # def test_always_fails(self):
#     #     self.assertTrue(False)
#


class FilePathsTesting(TestCase):
    def test_file_existence(self):
        from codes.common import common_datasets
        from pathlib import Path

        for dataset in common_datasets:
            filename = common_datasets[dataset].get("filename", False)
            if filename:
                example_file = Path(filename)
                self.assertTrue(example_file.exists())
