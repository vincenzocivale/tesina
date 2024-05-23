# Dizionario dei parametri per il fine tuning dei modelli
param_grids = {
    'rf': {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': [0.001, 0.01, 0.1],
        'clf__kernel': ['linear', 'rbf', 'poly'],
        'clf__probability': [True]
    },
    'gbm': {
        'clf__n_estimators': [50, 100, 150],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 4, 5]
    },
    'dt': {
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    'adaboost': {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.5]
    },
    'mlp': {
        'clf__hidden_layer_sizes': [(50,), (100,), (200,)],
        'clf__alpha': [0.0001, 0.001, 0.01]
    },
    'catboost': {
        'clf__iterations': [100, 200, 300],
        'clf__learning_rate': [0.01, 0.1, 0.2]
    }
}



#lista modelli senza parametri da ottimizzare
models_without_hyperparameters = ['nc', 'nb']