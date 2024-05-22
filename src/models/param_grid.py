# Dizionario dei parametri per il fine tuning dei modelli
param_grids = {
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1],
        'kernel': ['linear', 'rbf', 'poly'],
        'probability': [True]
    },
    'gbm': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    },
    'dt': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5]
    },
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (200,)],
        'alpha': [0.0001, 0.001, 0.01]
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}


#lista modelli senza parametri da ottimizzare
models_without_hyperparameters = ['nc', 'nb']