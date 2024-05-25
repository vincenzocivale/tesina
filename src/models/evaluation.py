from typing import List, Tuple, Dict, Any, Union
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef, accuracy_score
from sklearn.pipeline import Pipeline
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import traceback
from ..features import dimensionality_reduction as dr
from ..features import features_selection as fs
from . import param_grid as pg


def select_features_nested(X: pd.DataFrame, y: pd.Series, feature_selector: str, min_selections: int = 1, inner_folds: int = 3) -> List[str]:
    """
    Select features once for all models.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        feature_selector (str): The feature selection method to be used.
        min_selections (int, optional): The minimum number of times a feature must be selected to be included. Defaults to 1.
        inner_folds (int, optional): The number of folds for the inner cross-validation. Defaults to 3.

    Returns:
        List[str]: The selected features.
    """
    selected_features_counts = dict()

    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    for train_index, val_index in inner_cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        try:
            selected_features_fold = fs.select_features_with_statistic_test(X_train, y_train, feature_selector)
            for feature in selected_features_fold:
                selected_features_counts[feature] += 1
        except Exception as e:
            print(f"Error in feature selection for fold: {str(e)}")

    # Select features that were selected at least min_selections times
    selected_features_once = [feature for feature, count in selected_features_counts.items() if count >= min_selections]
    
    return selected_features_once

<<<<<<< HEAD


def evaluate_models(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], feature_selector: str, metric: Union[str, make_scorer] = 'f1_macro', logger: logging.Logger = None, outer_folds: int = 5, inner_folds: int = 3) -> Dict[str, Tuple[List[str], Dict[str, Any], float]]:
=======
def evaluate_models(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], feature_selector: str, metrics_list: List[Union[str, make_scorer]] = ['f1_macro', make_scorer(matthews_corrcoef, greater_is_better=True)]) -> Dict[str, Dict[str, Tuple[List[str], Dict[str, Any], float]]]:
>>>>>>> parent of 23fd83b (refactoring data organization functions)
    """
    Evaluate multiple models.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        models (Dict[str, Any]): A dictionary of machine learning models to be evaluated.
        feature_selector (str): The feature selection method to be used.
<<<<<<< HEAD
        metric (Union[str, make_scorer], optional): The metric to be used for evaluation. Defaults to 'f1_macro'.
        logger (logging.Logger, optional): The logger object to log messages. Defaults to None.
        outer_folds (int, optional): The number of folds for the outer cross-validation. Defaults to 5.
        inner_folds (int, optional): The number of folds for the inner cross-validation. Defaults to 3.
=======
        metrics_list (List[Union[str, make_scorer]], optional): The list of metrics to be used for evaluation. Defaults to ['f1_macro', make_scorer(matthews_corrcoef, greater_is_better=True)].
>>>>>>> parent of 23fd83b (refactoring data organization functions)

    Returns:
        Dict[str, Dict[str, Tuple[List[str], Dict[str, Any], float]]]: The results of the evaluation.
    """
<<<<<<< HEAD
    results = {}

    # Select features once for all models
    selected_features = select_features_nested(X, y, feature_selector, inner_folds=inner_folds)

    with tqdm(total=len(models), desc="Models Evaluation") as pbar:
        for model_name, model in models.items():
            pbar.set_description(f"Evaluating model: {model_name}")
            try:
                best_params, outer_scores = nested_cv_with_dim_reduction(X[selected_features], y, model, model_name, metric, logger, outer_folds=outer_folds, inner_folds=inner_folds)
                results[model_name] = (best_params, outer_scores)
            except Exception as e:
                if logger:
                    logger.error(f"Error evaluating model {model_name}: {str(e)}")
                else:
                    print(f"Error evaluating model {model_name}: {str(e)}")
            pbar.update(1)

    return results
=======
    metric_results = {}

    for metric in metrics_list:
        results = {}
        with ProcessPoolExecutor() as executor:
            future_to_model = {executor.submit(evaluate_model, X, y, model,  model_name, metric, feature_selector): model_name for model_name, model in models.items()}
            pbar = tqdm(total=len(future_to_model), desc=f"Models Evaluation with {metric}")

            for future in as_completed(future_to_model):
                name = future_to_model[future]
                try:
                    scores = future.result()
                    results[name] = scores
                except Exception as e:
                    print(f"Error evaluating model {name}: {str(e)}")
                pbar.update(1)
            pbar.close()
        metric_results[metric] = results
    return metric_results
>>>>>>> parent of 23fd83b (refactoring data organization functions)



def nested_cv_with_dim_reduction(X: pd.DataFrame, y: pd.Series, model: Any, model_name: str, scoring_metric: str, logger: logging.Logger, outer_folds: int = 5, inner_folds: int = 3) -> Tuple[List[Dict[str, Any]], List[List[str]], List[float]]:
    """
    Perform nested cross-validation with feature selection.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        model (Any): The machine learning model to be used.
        model_name (str): The name of the machine learning model.
        scoring_metric (str): The scoring metric to be used.
        logger (logging.Logger): The logger object to log messages.
        outer_folds (int, optional): The number of folds for the outer cross-validation. Defaults to 5.
        inner_folds (int, optional): The number of folds for the inner cross-validation. Defaults to 3.

    Returns:
        Tuple[List[Dict[str, Any]], List[List[str]], List[float]]: The best parameters, selected features, and outer scores.
    """
    param_grid = pg.param_grids[model_name]
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    best_params, selected_features, outer_scores = [], [], []

    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try:

            # Train model with selected features and evaluate on test set
            best_params_fold, outer_score = train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid, scoring_metric, logger)

            best_params.append(best_params_fold)
            outer_scores.append(outer_score)

            # Log the success
            logger.info(f"Nested CV for fold successful. Best params: {best_params_fold}, Score: {outer_score}")
        except Exception as e:
            error_details = traceback.format_exc()

            # Log the error
            logger.error(f"Error in nested CV for fold: {str(e)}")
            logger.error(f"Error details: {error_details}")

    return best_params, selected_features, outer_scores

def train_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, model: Any, param_grid: Dict[str, List[Any]], scoring_metric: str, logger: logging.Logger) -> Tuple[Dict[str, Any], float]:
    """
    Train and evaluate a model.

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training labels.
        X_test (pd.DataFrame): The test data.
        y_test (pd.Series): The test labels.
        model (Any): The machine learning model to be used.
        param_grid (Dict[str, List[Any]]): The parameter grid for hyperparameter tuning.
        scoring_metric (str): The scoring metric to be used.
        logger (logging.Logger): The logger object to log messages.

    Returns:
        Tuple[Dict[str, Any], float]: The best parameters found during training and the score on the test set.
    """
    pipeline = Pipeline([('reduce_dim', dr.dimensionality_red_dict['pca']), (model.__class__.__name__, model)])
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=scoring_metric)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    score = grid_search.score(X_test, y_test)

    # Log the success
    logger.info(f"Model training and evaluation successful. Best params: {best_params}, Score: {score}")

    return best_params, score
