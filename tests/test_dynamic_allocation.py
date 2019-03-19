import context
from src import dynamic_allocation
import pandas as pd
import numpy as np
import unittest
import os


class TestDynamicAllocation(unittest.TestCase):

    def setUp(self):
        self.model = dynamic_allocation.DynamicAllocation()

    def test_read_data(self):
        for file in os.listdir('./data'):
            df = pd.read_csv(os.path.join('data', file), sep=';', index_col='Date')
            self.assertIsInstance(df, pd.DataFrame)

    def test_set_initial_asset_weights(self):
        print(self.model._set_initial_asset_weights())
        self.assertEqual(len(self.model._prices_data().columns), 5)
        self.assertEqual(self.model._set_initial_asset_weights(), 0.2)

    def test_pct_change(self):
        df1 = pd.read_csv('data/pct_change.csv', sep=';', index_col='Date') / 100.0
        df2 = self.model._pct_change().round(4)
        pd.testing.assert_frame_equal(df1, df2)

    def test_partial_cummulative_return(self):
        df1 = self.model._partial_cummulative_return().round(4)
        df2 = pd.read_csv('data/partial_cumm_ret.csv', sep=';', index_col='Date') / 100.0
        self.assertAlmostEqual(df1, df2)

    def test_pre_allocation(self):
        pass

    def test_results(self):
        pass


if __name__ == '__main__':
    test = TestDynamicAllocation()
    test.test_read_data()
    #test._set_initial_asset_weights()