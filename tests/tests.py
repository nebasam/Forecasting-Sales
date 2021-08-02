import unittest
import sys
import os
from numpy import fix
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
sys.path.append(os.path.abspath(os.path.join('../scripts/')))
sys.path.insert(1, 'scripts')

import plotsClass

class ForecastSales(unittest.TestCase):

    def setUp(self):
        self.plot = plotsClass.Plot()
        self.Nulldf = pd.DataFrame({"A":[11, 5, None, 3, None, 8],
                   "B":[1, 5, None, 11, None, 8]})


    def test_fill_median(self):
        df = self.plot.fill_median(self.Nulldf, 'A')
        assert df['A'][2] == 6.5 and df['A'][4] == 6.5
    

    def tearDown(self) -> None:
        print('Closed')

if __name__ == '__main__':
    unittest.main()
    