# pylint: disable=import-error

import os

import mlflow


def get_model_location(run_id, experiment_idd):
    """
    Give model location on the provider
    """

    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlflow-artifact-remote-tex')
    experiment_id = experiment_idd

    model_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models'
    return model_location


def load_model(run_id, experiment_id_):
    model_path = get_model_location(run_id, experiment_id_)
    model = mlflow.pyfunc.load_model(model_path)
    return model
