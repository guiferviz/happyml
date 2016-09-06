
from unittest import TestCase
import numpy as np

import happyml.datasets as dt


DATASET01_CSV_NAME = "tests/fixtures/dataset01.csv"
DATASET01_TSV_NAME = "tests/fixtures/dataset01.tsv"
DATASET01 = dt.DataSet()
DATASET01.X = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
DATASET01.Y = np.array([[1], [1], [-1], [-1]])
DATASET01_ONE_HOT_Y = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]])


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

    def test_one_hot_output(self):
        dataset = dt.load(DATASET01_CSV_NAME, one_hot_output=True)
        np.testing.assert_array_equal(dataset.Y, DATASET01_ONE_HOT_Y,
                                      err_msg="DataSet Y not loaded correctly.")
        self.assertEqual(dataset.get_k(), 3)
        self.assertEqual(dataset.get_d(), 2)
        self.assertEqual(dataset.get_N(), 4)

    def test_dataset_get_type_bool_2(self):
        dataset = dt.DataSet()
        self.assertEqual(dataset.get_type(), "unknown")


class GetTypeTest(TestCase):

    def test_get_type_binary_1(self):
        y = np.array([[1], [0], [1]])
        self.assertEqual(dt.get_type(y), "binary")

    def test_get_type_binary_2(self):
        y = np.array([[1.0], [0.], [1]])
        self.assertEqual(dt.get_type(y), "binary")

    def test_get_type_binary_3(self):
        y = np.array([[1], [-1], [1]])
        self.assertEqual(dt.get_type(y), "binary")

    def test_get_type_binary_4(self):
        y = np.array([1, 4, 1])
        self.assertEqual(dt.get_type(y), "binary")

    def test_get_type_continuous_1(self):
        y = np.array([1.1, 4, 1])
        self.assertEqual(dt.get_type(y), "continuous")

    def test_get_type_continuous_2(self):
        y = np.array([[1.1], [1.1]])
        self.assertEqual(dt.get_type(y), "continuous")

    def test_get_type_continuous_3(self):
        y = np.array([[1, 2], [1.1, 2]])
        self.assertEqual(dt.get_type(y), "continuous-multioutput")

    def test_get_type_unknown_1(self):
        y = np.array([[[1], [2], [3]]])
        self.assertEqual(dt.get_type(y), "unknown")

    def test_get_type_unknown_2(self):
        y = np.empty((23, 0))
        self.assertEqual(dt.get_type(y), "unknown")

    def test_get_type_unknown_3(self):
        y = np.empty((0,))
        self.assertEqual(dt.get_type(y), "unknown")

    def test_get_type_unknown_4(self):
        y = 9
        self.assertEqual(dt.get_type(y), "unknown")

    def test_get_type_unknown_5(self):
        y = self
        self.assertEqual(dt.get_type(y), "unknown")

    def test_get_type_multiclass_1(self):
        y = np.array([[1], [2], [3]])
        self.assertEqual(dt.get_type(y), "multiclass")

    def test_get_type_multiclass_2(self):
        y = np.array([[1, 2, 1]])
        self.assertEqual(dt.get_type(y), "multiclass-multioutput")

    def test_get_type_multiclass_3(self):
        y = np.array([[1, 0, 1]])
        self.assertEqual(dt.get_type(y), "multiclass-multioutput")

    def test_get_type_multiclass_4(self):
        y = np.array([[1, 0], [0, 0]])
        self.assertEqual(dt.get_type(y), "multiclass-multioutput")

    def test_get_type_one_hot_1(self):
        y = np.array([[1, 0, 0]])
        self.assertEqual(dt.get_type(y), "multiclass-one-hot")

    def test_get_type_one_hot_3(self):
        y = np.array([[1, 0]])
        self.assertEqual(dt.get_type(y), "binary-one-hot")

    def test_get_type_str_1(self):
        y = np.array(["a", "b", "a"])
        self.assertEqual(dt.get_type(y), "binary")

    def test_get_type_str_2(self):
        y = np.array(["a", "b", "c"])
        self.assertEqual(dt.get_type(y), "multiclass")

    def test_get_type_str_3(self):
        y = np.array([["a", "b"]])
        self.assertEqual(dt.get_type(y), "multiclass-multioutput")

    def test_get_type_bool_1(self):
        y = np.array([True, False])
        self.assertEqual(dt.get_type(y), "binary")

    def test_get_type_bool_2(self):
        y = np.array([[True, False]])
        self.assertEqual(dt.get_type(y), "binary-one-hot")
