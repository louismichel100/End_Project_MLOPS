 1- Presentation of the project and the dataset

 Heart failure is responsible for nearly 30% of the death rate worldwide. So, we have chosen to set up a machine learning model that analyzes the data, the milkings and predicts the different heart failures that we could have. More specifically, we focused on arrhythmias and hypertrophies.

 Therefore, the model will take as input an ECG signal sampled at 300 or 500Hz, after signal amplification and noise elimination, will classify the signal according to the type of arrhythmia and hypertrophy.

 We used the electrocardiogram database of the University of Lobachevsky in Great Scotland downloadable from https://physionet.org/content/ludb/1.0.1/

 The path of the dataset in this project is ludb folder. Just check inside to look forward the cross analyse that had been done.

 For the simulation of the process, we choose to use an example of signal file : 'signal_heart_failure.csv

 We created an electrocardiogram with electrodes to be able to collect the signal of the patients and to be able to analyze it. This signal looks like this:

 ![signal](https://github.com/louismichel100/End_Project_MLOPS/blob/3e0da7c234c09298dd2281498dd24aa5d0824c2e/Capture%20d%E2%80%99%C3%A9cran%20de%202022-05-20%2017-20-36.png) 

 To understand how the pre-processing of the data was done, you must consult the jupyter file 'model_use.ipynb'

 run in a virtual environment 'pipenv install -r requirements.txt'

 Subsequently, we trained two models for predicting heart failure using the tensorflow and keras framework which requires cudart.dll for GPU execution. For the sake of confidentiality, we have voluntarily omitted certain technical aspects of the project because it is still in development and matches exactly what we have learned throughout this training

 2. HOW TO TEST THE PROJECT

    A- Build and run experiment tracking, model registry and prefect flow:

            model_use.ipynb and model_use.py

            python model_use.py

    B- Model deploy with AWS provider :

        python lambda_use_model.py

    C- Model deploy with docker

        // just look at the makefile

        make setup

    D- Model monitoring

        You need to run 'docker compose up' command to the root folder.
        Then, you run in another window 'run test_monitoring.py'
        You can an example of log in mongo db at test_mongo_db.ipynb file

    E- For tests around the project, just look the mak file or run :
    'pytest tests/'

    F- Look at terraform file main.tf to see infra

    ![signal](https://github.com/louismichel100/End_Project_MLOPS/blob/3e0da7c234c09298dd2281498dd24aa5d0824c2e/Capture%20d%E2%80%99%C3%A9cran%20du%202022-09-12%2022-52-04.png) 
