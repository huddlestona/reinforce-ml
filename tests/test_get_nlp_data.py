import pandas as pd
from scorer import Scorer
import datetime as dt
import numpy as np
import re
import unittest as unittest


class Test(unittest.TestCase):
    def __init__(self):
        self.path = '../data/180530_def.pkl'

    def test_get_nlp_data(self):
        data = pd.read_pickle(path)
        self.assertEqual(type(data), pandas.core.frame.DataFrame)
        scorer = Scorer()
        data['clean_def'] = data['definition'].apply(clean_def)
        data['nlp_doc'] = data['clean_def'].apply(
                            lambda x: add_nlp_doc(x, scorer))
        data['leaderboard'] = np.empty((len(data), 0)).tolist()
        self.assertEqual(type(data['leaderboard']), list)
        self.assertEqual(['leaderboard'.'clean_df','nlp_doc'] in data, True)
    

if __name__ == '__main__':
    unittest.main()

