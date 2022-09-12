 1- Presentation of the project and the dataset

 Heart failure is responsible for nearly 30% of the death rate worldwide. So, we have chosen to set up a machine learning model that analyzes the data, the milkings and predicts the different heart failures that we could have. More specifically, we focused on arrhythmias and hypertrophies.

 Therefore, the model will take as input an ECG signal sampled at 300 or 500Hz, after signal amplification and noise elimination, will classify the signal according to the type of arrhythmia and hypertrophy.

 We used the electrocardiogram database of the University of Lobachevsky in Great Scotland downloadable from https://physionet.org/content/ludb/1.0.1/

 The path of the dataset in this project is ludb folder. Just check inside to look forward the cross analyse that had been done.

 For the simulation of the process, we choose to use an example of signal file : 'signal_heart_failure.csv

 2. HOW TO TEST THE PROJECT

    A- Build and run experiment tracking and model registry:

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
