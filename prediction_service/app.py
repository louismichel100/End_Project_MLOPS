# pylint: disable=import-error

import os
import pickle

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from scipy import signal
from pymongo import MongoClient

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")


app = Flask('__name__')

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

##########


def filter_data(da, fs):

    """Filters the ECG data with a highpass at 0.1Hz and a bandstop around 50Hz (+/-2 Hz)"""

    b_dc, a_dc = signal.butter(4, (0.1 / fs * 2), btype='highpass')
    b_50, a_50 = signal.butter(4, [(48 / fs * 2), (52 / fs * 2)], btype='stop')

    da = signal.lfilter(b_dc, a_dc, da)
    da = signal.lfilter(b_50, a_50, da)

    return da


# Return difference array
def return_diff_array_table(array, dur):
    for idx in range(array.shape[1] - dur):
        before_col = array[:, idx]
        after_col = array[:, idx + dur]
        new_col = ((after_col - before_col) + 1) / 2
        new_col = new_col.reshape(-1, 1)
        if idx == 0:
            new_table = new_col
        else:
            new_table = np.concatenate((new_table, new_col), axis=1)
    # For concat add zero padding
    padding_array = np.zeros(shape=(array.shape[0], dur))
    new_table = np.concatenate((padding_array, new_table), axis=1)
    return new_table


# For plotting the signal heart
def plot_(kp):

    np_count = np.linspace(5, 186, 187)
    np_time = np.tile(np_count, (1, 1))
    a = kp.iloc[2, np_time[0, 0:100]]
    a = pd.concat([a, kp.iloc[3, np_time[0, 0:100]]], axis=0)
    a = pd.concat([a, kp.iloc[4, np_time[0, 0:140]]], axis=0)
    a.reset_index(inplace=True, drop=True)
    ax = plt.subplot(2, 3, 1)
    ax.set_title("Cycle cardiaque")
    ax.plot(a)
    plt.savefig("./static/signal_show")


# Concat
def return_merge_diff_table(df, diff_dur):
    fin_table = df.reshape(-1, 187, 1, 1)
    for dur in diff_dur:
        temp_table = return_diff_array_table(df, dur)
        fin_table = np.concatenate(
            (fin_table, temp_table.reshape(-1, 187, 1, 1)), axis=2
        )
    return fin_table


def predict_endpoint(features):
    # features = intensityNormalisationFeatureScaling(features, float)
    with open('model_a1.bin', 'rb') as f_in:
        model_n = pickle.load(f_in)

    features = filter_data(features, 300)
    X = return_merge_diff_table(features, diff_dur=[1])
    preds = model_n.predict(X)
    return preds


def next_endpoint(features):
    with open('model_ght.bin', 'rb') as f_in:
        model_2 = pickle.load(f_in)

    # features = intensityNormalisationFeatureScaling(features, float)
    features = filter_data(features, 300)
    X = features.reshape(len(features), features.shape[1], 1)
    preds = model_2.predict(X)
    return preds


@app.route('/predict', methods=['POST'])
def predict():

    record = request.get_json()
    print(record)
    debut = int(record['debut_signal'])
    numero = int(record['numero_personne'])
    df = pd.read_csv('signal_heart_failure.csv', header=None)

    df_2 = df.iloc[numero : numero + 1, debut : debut + 187]
    # df_3 = df.iloc[numero : numero + 1, debut : debut + 186]
    # plot_(df)

    features_t = np.array(df_2)
    # features_tt = np.array(df_3)

    pred = predict_endpoint(features_t)

    # print(pred[0])
    chan = {
        0.0: 'Normal Beat',
        1.0: 'Supraventricular premature beat',
        2.0: 'Premature ventricular contraction',
        3.0: 'Fusion of ventricular and normal beat',
        4.0: 'Unclassifiable beat',
    }

    res1 = ""
    disease = ""

    if float(pred[0][0]) > float(pred[0][1]):
        res1 = "healthy"
        disease = "nothing"
    else:
        disease = chan[1]
        res1 = "sickness"

    result = {'decision': res1, 'exam': disease}

    print(result)

    save_to_db(record, disease)
    send_to_evidently_service(record, disease)
    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/sickness", json=[rec])


if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(debug=True)
