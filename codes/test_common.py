from unittest import TestCase

import numpy as np
import pandas as pd

from codes.common import define_batches, find_indexes, load_and_prepare_dataset


class CommonTesting(TestCase):
    def test_define_batches_true(self):
        df = pd.DataFrame(np.random.randint(0, 10, size=(100, 2)))
        df_with_batches = define_batches(df, 10)
        self.assertTrue(df_with_batches["Batch"].unique().shape[0] == 10)

    def test_define_batches_false(self):
        df = pd.DataFrame(np.random.randint(0, 10, size=(105, 2)))
        df_with_batches = define_batches(df, 10)
        self.assertFalse(df_with_batches["Batch"].unique().shape[0] == 10)

    def test_find_indexes(self):
        drifts_list = [None, None, "drift", None, "drift"]
        drift_indexes = find_indexes(drifts_list)
        self.assertEqual(drift_indexes, [2, 4])

    def test_load_prepare_insect_dataset(self):
        df, _, dataset_id = load_and_prepare_dataset("Abrupt (imbal.)")
        self.assertFalse(df.empty)
        self.assertEqual(dataset_id, "abrupt_imbal")

    def test_load_prepare_magic_dataset(self):
        df, _, dataset_id = load_and_prepare_dataset("magic")
        self.assertFalse(df.empty)
        self.assertEqual(dataset_id, "magic")

    def test_load_prepare_electricity_dataset(self):
        df, _, dataset_id = load_and_prepare_dataset("electricity")
        self.assertFalse(df.empty)
        self.assertEqual(dataset_id, "electricity")

    def test_load_prepare_sea_dataset(self):
        df, _, dataset_id = load_and_prepare_dataset("SEA")
        self.assertFalse(df.empty)
        self.assertEqual(dataset_id, "SEA")

    def test_load_prepare_stagger_dataset(self):
        df, _, dataset_id = load_and_prepare_dataset("STAGGER")
        self.assertFalse(df.empty)
        self.assertEqual(dataset_id, "STAGGER")

    # def test_load_synthetic_dataset(self):
    #     params = {
    #         "dataset_size": 3000,
    #         "drift_kind": "abrupt",
    #     }
    #     df, dataset_id = load_and_prepare_dataset("synthetic", params)
    #     self.assertFalse(df.empty)
    #     self.assertEqual(dataset_id, "synthetic")
