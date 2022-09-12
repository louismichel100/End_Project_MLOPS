# pylint: disable=unused-variable
# pylint: disable=unspecified-encoding

#!/usr/bin/env python
# coding: utf-8

import os
import itertools

import wfdb
import numpy as np
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from prefect import flow, task
from keras.layers import Dense, Input, Flatten, MaxPool1D, Convolution1D
from keras.models import Model
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# from prefect.storage import S3

path = 'ludb'
save_path = ''
fs_out = 500
test_ratio = 0.2


@task
def plot_confusion_matrix(
    cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


@task
def valt(mp):
    return (len(mp), mp[10], mp[11], mp[12])


@task
def drop_nan(data):
    j = 0
    for x in data:
        if pd.isnull(x):
            data[j] = 0
        j = j + 1


@task
def drop_zero(data):
    j = 0
    ms = data.mean()
    for x in data:
        if data[j] == 0.0:
            data[j] = x
            data[j] = ms
        j = j + 1


@task
def network(X_train, y_train, X_test, y_test):

    im_shape = (X_train.shape[1], 1)
    inputs_cnn = Input(shape=(im_shape), name='inputs_cnn')
    conv1_1 = Convolution1D(128, (3), activation='relu', input_shape=im_shape)(
        inputs_cnn
    )
    pool1 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1 = Convolution1D(128, (3), activation='relu', input_shape=im_shape)(pool1)
    pool2 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv2_1)

    conv3_1 = Convolution1D(128, (3), activation='relu', input_shape=im_shape)(pool2)
    pool3 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv3_1)

    conv4_1 = Convolution1D(128, (3), activation='relu', input_shape=im_shape)(pool3)
    pool4 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv4_1)

    flatten = Flatten()(pool4)
    dense_end1 = Dense(128, activation='relu')(flatten)
    main_output = Dense(2, activation='sigmoid', name='main_output')(dense_end1)

    model = Model(inputs=inputs_cnn, outputs=main_output)
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
    )

    # callbacks = [EarlyStopping(monitor='val_loss', patience=10),
    #         ModelCheckpoint(filepath='md.bin', monitor='val_loss', save_best_only=True)]

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        steps_per_epoch=100,
        validation_data=(X_test, y_test),
    )

    # model.load_weights('md.h5')
    return model


@task
def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print(f"Accuracy: {scores[1] * 100}")

    print(history)
    # fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("ac_t2", dpi=300)
    plt.show()

    # fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig("ac_t_22", dpi=300)
    plt.show()
    # target_names=['0','1','2','3','4']

    # Compute confusion matrix


@task
def get_dict():

    rhy_dict = {
        'Sinus rhythm': 1.0,
        'Sinus tachycardia': 2.0,
        'Sinus bradycardia': 3.0,
        'Sinus arrhythmia': 4.0,
        'Irregular sinus rhythm': 5.0,
        'Atrial fibrillation': 6.0,
        'Atrial flutter, typical': 7.0,
        'Abnormal rhythm': 8.0,
    }

    elec_dict = {
        'Electric axis of the heart: normal': 1.0,
        'Electric axis of the heart: left axis deviation': 2.0,
        'Electric axis of the heart: vertical': 3.0,
        'Electric axis of the heart: horizontal': 4.0,
        'Electric axis of the heart: right axis deviation': 5.0,
        0.0: 0.0,
    }

    con_dict = {
        'Sinoatrial blockade, undefined': 1.0,
        'I degree av block': 2.0,
        'Iii degree av-block': 3.0,
        'Incomplete right bundle branch block': 4.0,
        'Incomplete left bundle branch block': 5.0,
        'Left anterior hemiblock': 6.0,
        'Complete right bundle branch block': 7.0,
        'Complete left bundle branch block': 8.0,
        'Non-specific intravintricular conduction delay': 9.0,
        0.0: 0.0,
    }

    ext_dict = {
        'Atrial extrasystole: undefined': 1.0,
        'Atrial extrasystole: low atrial': 2.0,
        'Atrial extrasystole: left atrial': 3.0,
        'Atrial extrasystole, SA-nodal extrasystole': 4.0,
        'Atrial extrasystole, type: single pac': 5.0,
        'Atrial extrasystole, type: bigemini': 6.0,
        'Atrial extrasystole, type: quadrigemini': 7.0,
        'Atrial extrasystole, type: allorhythmic pattern': 8.0,
        'Ventricular extrasystole, morphology: polymorphic': 9.0,
        'Ventricular extrasystole, localisation: rvot, anterior wall': 10.0,
        'Ventricular extrasystole, localisation: rvot, antero-septal part': 11.0,
        'Ventricular extrasystole, localisation: IVS, middle part': 12.0,
        'Ventricular extrasystole, localisation: LVOT, LVS': 13.0,
        'Ventricular extrasystole, localisation: LV, undefined': 14.0,
        'Ventricular extrasystole, type: single pvc': 15.0,
        'Ventricular extrasystole, type: intercalary pvc': 16.0,
        'Ventricular extrasystole, type: couplet': 17.0,
        0.0: 0.0,
    }

    hyp_dict = {
        'Right atrial hypertrophy': 1.0,
        'Left atrial hypertrophy': 2.0,
        'Right atrial overload': 3.0,
        'Left atrial overload': 4.0,
        'Left ventricular hypertrophy': 5.0,
        'Right ventricular hypertrophy': 6.0,
        'Left ventricular overload': 7.0,
        0.0: 0.0,
    }

    card_dict = {'Pacemaker presence, undefined': 1.0, 'P-synchrony': 2.0, 0.0: 0.0}

    isch_dict = {
        'Stemi: anterior wall': 1.0,
        'Stemi: lateral wall': 2.0,
        'Stemi: septal': 3.0,
        'Stemi: inferior wall': 4.0,
        'Stemi: apical': 5.0,
        'Ischemia: anterior wall': 6.0,
        'Ischemia: lateral wall': 7.0,
        'Ischemia: septal': 8.0,
        'Ischemia: inferior wall': 9.0,
        'Ischemia: posterior wall': 10.0,
        'Ischemia: apical': 11.0,
        'Scar formation: lateral wall': 12.0,
        'Scar formation: septal': 13.0,
        'Scar formation: inferior wall': 14.0,
        'Scar formation: posterior wall': 15.0,
        'Scar formation: apical': 16.0,
        'Undefined ischemia/scar/supp.NSTEMI: anterior wall': 17.0,
        'Undefined ischemia/scar/supp.nstemi: lateral wall': 18.0,
        'Undefined ischemia/scar/supp.NSTEMI: septal': 19.0,
        'Undefined ischemia/scar/supp.nstemi: inferior wall': 20.0,
        'Undefined ischemia/scar/supp.nstemi: posterior wall': 21.0,
        'Undefined ischemia/scar/supp.nstemi: apical': 22.0,
        0.0: 0.0,
    }

    nons_dict = {
        'Non-specific repolarization abnormalities: inferior wall': 1.0,
        'Non-specific repolarization abnormalities: lateral wall': 2.0,
        'Non-specific repolarization abnormalities: anterior wall': 3.0,
        'Non-specific repolarization abnormalities: posterior wall': 4.0,
        'Non-specific repolarization abnormalities: apical': 5.0,
        'Non-specific repolarization abnormalities: septal': 6.0,
        0.0: 0.0,
    }

    oth_dict = {'Early repolarization syndrome': 1.0, 0.0: 0.0}

    sex_dict = {'M': 1.0, 'F': 0.0}

    return (
        rhy_dict,
        elec_dict,
        con_dict,
        ext_dict,
        hyp_dict,
        card_dict,
        isch_dict,
        nons_dict,
        oth_dict,
        sex_dict,
    )


@task
def get_signal():
    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    data_files = [
        "ludb/data/" + file for file in os.listdir("ludb/data/") if ".dat" in file
    ]
    data_files = sorted(data_files, key=valt)
    chanel = []
    # signal_out = []
    signal_all = []

    for miki in data_files:
        # signal_out = []
        for i in range(12):
            chanel = []
            chanel.append(i)
            sig = wfdb.rdsamp(miki[:-4], channels=chanel)
            sig_1d = np.ravel(sig[0])
            signal_all.append(sig_1d)
    df = pd.DataFrame(signal_all)

    return df, all_record_name


@task
def get_weith():
    (
        rhy_dict,
        elec_dict,
        con_dict,
        ext_dict,
        hyp_dict,
        card_dict,
        isch_dict,
        nons_dict,
        oth_dict,
        sex_dict,
    ) = get_dict()

    df_2 = pd.read_csv(path + '/ludb.csv')

    df_2['Age'] = df_2['Age'].str.split('\n').str[0]
    df_2['Sex'] = df_2['Sex'].str.split('\n').str[0]
    df_2['Ischemia'] = df_2['Ischemia'].str.split('\n').str[0]
    df_2['Cardiac pacing'] = df_2['Cardiac pacing'].str.split('\n').str[0]
    df_2['Extrasystolies'] = df_2['Extrasystolies'].str.split('\n').str[0]
    df_2['Non-specific repolarization abnormalities'] = (
        df_2['Non-specific repolarization abnormalities'].str.split('\n').str[0]
    )
    df_2['Hypertrophies'] = df_2['Hypertrophies'].str.split('\n').str[0]
    df_2['Electric axis of the heart'] = (
        df_2['Electric axis of the heart'].str.split('\n').str[0]
    )
    df_2['Rhythms'] = df_2['Rhythms'].str.split('\n').str[0]
    df_2['Conduction abnormalities'] = (
        df_2['Conduction abnormalities'].str.split('\n').str[0]
    )
    df_2['Other states'] = df_2['Other states'].str.split('\n').str[0]

    drop_nan(df_2['Conduction abnormalities'])
    drop_nan(df_2['Extrasystolies'])
    drop_nan(df_2['Hypertrophies'])
    drop_nan(df_2['Cardiac pacing'])
    drop_nan(df_2['Ischemia'])
    drop_nan(df_2['Non-specific repolarization abnormalities'])
    drop_nan(df_2['Other states'])
    drop_nan(df_2['Electric axis of the heart'])

    df_2['Rhythms'] = [rhy_dict[item] for item in df_2['Rhythms']]
    df_2['Electric axis of the heart'] = [
        elec_dict[item] for item in df_2['Electric axis of the heart']
    ]
    df_2['Conduction abnormalities'] = [
        con_dict[item] for item in df_2['Conduction abnormalities']
    ]
    df_2['Extrasystolies'] = [ext_dict[item] for item in df_2['Extrasystolies']]
    df_2['Hypertrophies'] = [hyp_dict[item] for item in df_2['Hypertrophies']]
    df_2['Cardiac pacing'] = [card_dict[item] for item in df_2['Cardiac pacing']]
    df_2['Ischemia'] = [isch_dict[item] for item in df_2['Ischemia']]
    df_2['Non-specific repolarization abnormalities'] = [
        nons_dict[item] for item in df_2['Non-specific repolarization abnormalities']
    ]
    df_2['Other states'] = [oth_dict[item] for item in df_2['Other states']]
    df_2['Sex'] = [sex_dict[item] for item in df_2['Sex']]

    df_2['Age'][df_2['ID'] == 34] = '89'
    df_2['Age'] = df_2['Age'].astype('float')

    drop_zero(df_2['Conduction abnormalities'])
    drop_zero(df_2['Extrasystolies'])
    drop_zero(df_2['Hypertrophies'])
    drop_zero(df_2['Cardiac pacing'])
    drop_zero(df_2['Ischemia'])
    drop_zero(df_2['Non-specific repolarization abnormalities'])
    drop_zero(df_2['Other states'])
    drop_zero(df_2['Electric axis of the heart'])

    df_n = []
    for j in range(200):
        dd = df_2.iloc[j : j + 1, :]
        for i in range(12):
            df_n.append(dd)

    a = np.reshape(df_n, (2400, 12))
    df_fn2 = pd.DataFrame(a)
    df_fn2.columns = [
        'ID',
        'Sex',
        'Age',
        'Rhythms',
        'Electric axis of the heart',
        'Conduction abnormalities',
        'Extrasystolies',
        'Hypertrophies',
        'Cardiac pacing',
        'Ischemia',
        'Non-specific repolarization abnormalities',
        'Other states',
    ]
    print(df_fn2.shape)

    return df_fn2


@task
def get_param(lenght):

    df, all_record_name = get_signal()
    df_fn2 = get_weith()

    df_all = pd.concat([df, df_fn2], axis=1)
    all_record_new = []

    for code in all_record_name:
        all_record_new.append(int(code[5:]))
    predictors = df_all

    ps = ['Rhythms', 'Electric axis of the heart']
    target = df_all[ps]  # Strength column
    predictors.drop(
        [
            'Rhythms',
            'Electric axis of the heart',
            'Conduction abnormalities',
            'Extrasystolies',
            'Hypertrophies',
            'Cardiac pacing',
            'Ischemia',
            'Non-specific repolarization abnormalities',
            'Other states',
        ],
        1,
        inplace=True,
    )
    predictors.drop('ID', 1, inplace=True)
    predictors.drop('Age', 1, inplace=True)
    predictors.drop('Sex', 1, inplace=True)

    predictors = predictors.iloc[:, 0 : int(lenght)]

    X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test


@task
def set_new_experiment(name, lenght, epoch, step_per_epoch):
    X_train, X_test, y_train, y_test = get_param(lenght=lenght)
    mlflow.set_experiment(name)

    with mlflow.start_run():

        params = {"epoch": int(epoch), "step_per_epoch": int(step_per_epoch)}
        mlflow.log_params(params)
        mlflow.autolog()

        model = network(X_train, y_train, X_test, y_test)
        y_pred = np.array(model.predict(X_test))
        y_test_ = np.array(y_test)
        mlflow.log_metric(
            "accuracy", accuracy_score(y_test_.argmax(axis=1), y_pred.argmax(axis=1))
        )

        mlflow.sklearn.log_model(model, artifact_path="models")
        # print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")


@flow()
def main():

    os.environ[
        "AWS_PROFILE"
    ] = "tex-profile"  # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials

    TRACKING_SERVER_HOST = "ec2-34-208-210-149.us-west-2.compute.amazonaws.com"  # fill in with the public DNS of the EC2 instance
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    set_new_experiment("my_experiment_9", 186, 5, 100)

    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

    client = MlflowClient(f"http://{TRACKING_SERVER_HOST}:5000")

    run_id = client.list_run_infos(experiment_id='9')[0].run_id
    mlflow.register_model(model_uri=f"runs:/{run_id}/models", name='signal_186')

    print(client.list_registered_models())

    print("voici le run id")
    print(run_id)


main()
