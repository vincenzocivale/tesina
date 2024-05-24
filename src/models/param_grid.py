# Dizionario dei parametri per il fine tuning dei modelli

param_grids = {
    'svm': {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.001, 0.01, 0.1],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__probability': [True]
    },
    'gbm': {
        'gbm__n_estimators': [50, 100, 150],
        'gbm__learning_rate': [0.01, 0.1, 0.2],
        'gbm__max_depth': [3, 4, 5]
    },
    'dt': {
        'dt__max_depth': [None, 10, 20],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 4]
    },
    'adaboost': {
        'adaboost__n_estimators': [50, 100, 200],
        'adaboost__learning_rate': [0.01, 0.1, 0.5]
    },
    'mlp': {
        'mlp__hidden_layer_sizes': [(50,), (100,), (200,)],
        'mlp__alpha': [0.0001, 0.001, 0.01]
    },
    'catboost': {
        'catboost__iterations': [100, 200, 300],
        'catboost__learning_rate': [0.01, 0.1, 0.2]
    },
    'rf': {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }
}





#lista modelli senza parametri da ottimizzare
models_without_hyperparameters = ['nc', 'nb']