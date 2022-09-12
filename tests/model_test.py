import numpy as np
import pandas as pd

import lambda_use_model
import prediction_service.app as po


def test_filter_data():
    df1 = pd.read_csv('signal_heart_failure.csv', header=None)

    df1 = df1.iloc[3:4, 0:186]

    features_tt = np.array(df1)

    actual_features = lambda_use_model.filter_data(features_tt, 300)[0][0]

    expected_features = 2.681509651192509

    assert actual_features == expected_features


def test_return_merge_diff_table():
    df = pd.read_csv('signal_heart_failure.csv', header=None)

    numero = 2
    debut = 0
    features_t = np.array(df.iloc[numero : numero + 1, debut : debut + 187])

    actual_features = po.return_merge_diff_table(features_t, [1])[0][0][0][0]

    expected_features = 2.0

    assert actual_features == expected_features
