from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from . import param_grid as pg

# Definisce modelli da valutare
def define_models(enable_categorical=True):
    """
    Define a dictionary of machine learning models for evaluation.

    Args:
        enable_categorical (bool): Whether to enable categorical features (default is True).

    Returns:
        dict: A dictionary where keys are model names and values are model objects.
    """
    models = {
        'svm': SVC(),
        'dt': DecisionTreeClassifier(),
        'gbm': GradientBoostingClassifier(),
        'adaboost': AdaBoostClassifier(),
        'mlp': MLPClassifier(),
        'catboost': CatBoostClassifier(logging_level='Silent'),
        'rf': RandomForestClassifier(n_estimators=100)
    }
    return models

# Crea pipeline per il modello
def make_pipeline(model):
    """
    Create a pipeline for a given machine learning model.

    Args:
        model: The machine learning model to be included in the pipeline.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    """
    return Pipeline(steps=[('model', model)])

# Funzione per il fine tuning del modello
def fine_tune_model(model_name, model, X, y, metric, cv):
    
    if model_name in pg.param_grids:
        param_grid = pg.param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, scoring=metric, cv=cv, n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_params_
    else:
        if model_name in pg.models_without_hyperparameters:

           print(f"Model {model_name} does not have hyperparameters for fine tuning")
           return None
        
        raise ValueError("Model name not supported")
    
def train_model_with_optimal_params(model_name, optimal_parameters, X_train, y_train):
    """
    Train a machine learning model with optimal parameters.

    Args:
        model_name (str): The name of the model.
        optimal_parameters (dict): A dictionary containing the optimal parameters for the model.
        X_train (array-like): Training input data.
        y_train (array-like): Training target labels.

    Returns:
        trained_model: The trained machine learning model with optimal parameters.
    """

    models = define_models()
    
    if model_name not in optimal_parameters:
        raise ValueError(f"Model '{model_name}' not found in optimal parameters.")

    model = models[model_name].set_params(**optimal_parameters[model_name])
    trained_model = model.fit(X_train, y_train)

    return trained_model
