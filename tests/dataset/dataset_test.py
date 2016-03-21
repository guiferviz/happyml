
from unittest import TestCase
import numpy as np

import happyml.dataset as dt


DATASET01_CSV_NAME = "tests/fixtures/dataset01.csv"
DATASET01_TSV_NAME = "tests/fixtures/dataset01.tsv"
DATASET01 = dt.DataSet()
DATASET01.X = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
DATASET01.Y = np.array([[1], [1], [-1], [-1]])
DATASET01_ONE_SHOT_Y = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]])


class DataSetTest(TestCase):

    def test_get_N(self):
        self.assertEqual(DATASET01.get_N(), 4)

    def test_get_d(self):
        self.assertEqual(DATASET01.get_d(), 2)

    def test_get_k(self):
        self.assertEqual(DATASET01.get_k(), 1)

    def test_load_csv(self):
        dataset = dt.load(DATASET01_CSV_NAME)
        np.testing.assert_array_equal(dataset.X, DATASET01.X,
                                      err_msg="DataSet X not loaded correctly.")
        np.testing.assert_array_equal(dataset.Y, DATASET01.Y,
                                      err_msg="DataSet Y not loaded correctly.")

    def test_load_tsv(self):
        dataset = dt.load(DATASET01_TSV_NAME)
        np.testing.assert_array_equal(dataset.X, DATASET01.X,
                                      err_msg="DataSet X not loaded correctly.")
        np.testing.assert_array_equal(dataset.Y, DATASET01.Y,
                                      err_msg="DataSet Y not loaded correctly.")

    def test_n_outputs_0(self):
        dataset = dt.load(DATASET01_CSV_NAME, n_outputs=2)
        self.assertEqual(dataset.get_k(), 2)
        self.assertEqual(dataset.get_d(), 1)
        self.assertEqual(dataset.get_N(), 4)

    def test_n_outputs_1(self):
        dataset = dt.load(DATASET01_CSV_NAME, n_outputs=3)
        self.assertEqual(dataset.get_k(), 3)
        self.assertEqual(dataset.get_d(), 0)
        self.assertEqual(dataset.get_N(), 4)

    def test_n_outputs_2(self):
        with self.assertRaises(AssertionError):
            dt.load(DATASET01_CSV_NAME, n_outputs=4)

    def test_n_outputs_3(self):
        dataset = dt.load(DATASET01_CSV_NAME, n_outputs=-1)
        self.assertEqual(dataset.get_k(), 1)
        self.assertEqual(dataset.get_d(), 2)
        self.assertEqual(dataset.get_N(), 4)

    def test_n_outputs_4(self):
        dataset = dt.load(DATASET01_CSV_NAME, n_outputs=-2)
        self.assertEqual(dataset.get_k(), 2)
        self.assertEqual(dataset.get_d(), 1)
        self.assertEqual(dataset.get_N(), 4)

    def test_n_outputs_5(self):
        dataset = dt.load(DATASET01_CSV_NAME, n_outputs=-3)
        self.assertEqual(dataset.get_k(), 3)
        self.assertEqual(dataset.get_d(), 0)
        self.assertEqual(dataset.get_N(), 4)

    def test_n_outputs_6(self):
        with self.assertRaises(AssertionError):
            dt.load(DATASET01_CSV_NAME, n_outputs=-4)

    def test_not_found_file(self):
        with self.assertRaises(IOError):
            dt.load("happy.csv")

    def test_one_shot_output(self):
        dataset = dt.load(DATASET01_CSV_NAME, one_shot_output=True)
        np.testing.assert_array_equal(dataset.Y, DATASET01_ONE_SHOT_Y,
                                      err_msg="DataSet Y not loaded correctly.")
        self.assertEqual(dataset.get_k(), 3)
        self.assertEqual(dataset.get_d(), 2)
        self.assertEqual(dataset.get_N(), 4)
