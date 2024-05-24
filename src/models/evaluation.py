from typing import List, Tuple, Dict, Any, Union
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef, accuracy_score
from sklearn.pipeline import Pipeline
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
from ..features import dimensionality_reduction as dr
from ..features import features_selection as fs
from . import param_grid as pg



def nested_cv_with_dim_reduction(X: pd.DataFrame, y: pd.Series, model: Any, model_name: str, feature_selector: str, dim_reduction: Any, scoring_metric: str,  outer_folds: int = 5, inner_folds: int = 3) -> Tuple[List[Dict[str, Any]], List[List[str]], List[float]]:
    """
    Perform a nested cross-validation, applying dimensionality reduction.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        model (Any): The machine learning model to be used.
        model_name (str): The name of the machine learning model.
        feature_selector (str): The feature selection method to be used.
        dim_reduction (Any): The dimensionality reduction method to be used.
        scoring_metric (str): The scoring metric to be used.
        outer_folds (int, optional): The number of folds for the outer cross-validation. Defaults to 5.
        inner_folds (int, optional): The number of folds for the inner cross-validation. Defaults to 3.

    Returns:
        Tuple[List[Dict[str, Any]], List[List[str]], List[float]]: The best parameters, selected features, and outer scores.
    """
    param_grid = pg.param_grids[model_name]
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    best_params, selected_features, outer_scores = [], [], []

    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try:
            features_selected = fs.select_features_with_statistic_test(X_train, y_train, feature_selector)
            X_train, X_test = X_train.loc[:, features_selected], X_test.loc[:, features_selected]
            selected_features.append(features_selected)

            pipeline = Pipeline([('reduce_dim', dim_reduction), (model_name, model)])
            grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=scoring_metric)
            grid_search.fit(X_train, y_train)

            best_params.append(grid_search.best_params_)
            outer_scores.append(grid_search.score(X_test, y_test))
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in nested CV for fold: {str(e)}")
            print(f"Error details: {error_details}")

    return best_params, selected_features, outer_scores

def evaluate_models(X: pd.DataFrame, y: pd.Series, models: Dict[str, Any], feature_selector: str, metrics_list: List[Union[str, make_scorer]] = ['f1_macro', make_scorer(matthews_corrcoef, greater_is_better=True)]) -> Dict[str, Dict[str, Tuple[List[str], Dict[str, Any], float]]]:
    """
    Evaluate multiple models.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        models (Dict[str, Any]): A dictionary of machine learning models to be evaluated.
        feature_selector (str): The feature selection method to be used.
        metrics_list (List[Union[str, make_scorer]], optional): The list of metrics to be used for evaluation. Defaults to ['f1_macro', make_scorer(matthews_corrcoef, greater_is_better=True)].

    Returns:
        Dict[str, Dict[str, Tuple[List[str], Dict[str, Any], float]]]: The results of the evaluation.
    """
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

def evaluate_model(X: pd.DataFrame, y: pd.Series, model: Any, model_name: str, scoring_metric: str, feature_selector_key: str,  dim_reduction_key: str = 'pca', outer_folds: int = 5, inner_folds: int = 3) -> Tuple[List[str], Dict[str, Any], float]:
    """
    Evaluate a single model.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        model (Any): The machine learning model to be evaluated.
        model_name (str): The name of the machine learning model.
        scoring_metric (str): The scoring metric to be used.
        feature_selector_key (str): The feature selection method to be used.
        dim_reduction_key (str, optional): The dimensionality reduction method to be used. Defaults to 'pca'.
        outer_folds (int, optional): The number of folds for the outer cross-validation. Defaults to 5.
        inner_folds (int, optional): The number of folds for the inner cross-validation. Defaults to 3.

    Returns:
        Tuple[List[str], Dict[str, Any], float]: The most common features, most common parameters, and score.
    """
    dim_reduction = dr.dimensionality_red_dict.get(dim_reduction_key)
    if dim_reduction is None:
        raise ValueError(f"Invalid dimensionality reduction key: {dim_reduction_key}")

    try:
        best_params, selected_features, outer_scores = nested_cv_with_dim_reduction(X, y, model, model_name, feature_selector_key, dim_reduction, scoring_metric, outer_folds=outer_folds, inner_folds=inner_folds)

        feature_counts = Counter([feature for sublist in selected_features for feature in sublist])
        most_common_features = [feature for feature, count in feature_counts.most_common() if count > 1]

        param_counts = Counter([str(param) for param in best_params])
        most_common_params = eval(param_counts.most_common(1)[0][0])

        pipeline = Pipeline([('reduce_dim', dim_reduction), (model_name, model)])
        pipeline.set_params(**most_common_params)
        pipeline.fit(X[most_common_features], y)

        score = evaluate_score(X, y, model, most_common_features, scoring_metric)

        return most_common_features, most_common_params, score
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        return None, None, None

def evaluate_score(X: pd.DataFrame, y: pd.Series, model: Any, features: List[str], scoring_metric: str) -> float:
    """
    Evaluate the model performance on the entire dataset.

    Args:
        X (pd.DataFrame): The input data.
        y (pd.Series): The target labels.
        model (Any): The machine learning model.
        features (List[str]): List of selected features.
        scoring_metric (str): The evaluation metric to be used.

    Returns:
        float: The score obtained by the model.
    """
    try:
        y_pred = model.predict(X[features])
        if scoring_metric == 'accuracy':
            return accuracy_score(y, y_pred)
        elif scoring_metric == 'f1_macro':
            return f1_score(y, y_pred, average='macro')
        elif scoring_metric == 'matthews_corrcoef':
            return matthews_corrcoef(y, y_pred)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring_metric}")
    except Exception as e:
        print(f"Error in calculate model performance: {str(e)}")
        return None
