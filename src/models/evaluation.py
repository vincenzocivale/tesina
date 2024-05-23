from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from statistics import mean, stdev
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from tqdm import tqdm
from collections import Counter
from  ..features import dimensionality_reduction as dr
from ..features import features_selection as fs
from . import param_grid as pg

def summarize_results(results, top_n=10, ascending=False):
    """
    Summarize the evaluation results of machine learning models.

    Args:
        results (dict): Dictionary containing evaluation results where keys are evaluation metrics and values are dictionaries of model names and corresponding scores.
        top_n (int): Number of top models to be included in the summary (default is 10).
        ascending (bool): Whether to sort scores in ascending order (default is False).

    Returns:
        None
    """
    for metric, result in results.items():
        print(f"\nMetric: {metric}")
        sorted_results = sorted(result.items(), key=lambda x: mean(x[1]), reverse=not ascending)
        for rank, (name, scores) in enumerate(sorted_results[:top_n], 1):
            mean_score, std_score = mean(scores), stdev(scores)
            print(f"Rank={rank}, Name={name}, Score={mean_score:.3f} (+/- {std_score:.3f})")
            print(f"    Scores: {scores}")


# Trova i migliori modelli
def find_top_models(results, top_n=3, ascending=False):
    """
    Summarize the evaluation results of machine learning models and return the top N model names for each metric.

    Args:
        results (dict): Dictionary containing evaluation results where keys are evaluation metrics and values are dictionaries of model names and corresponding scores.
        top_n (int): Number of top models to be included in the summary (default is 3).
        ascending (bool): Whether to sort scores in ascending order (default is False).

    Returns:
        dict: Dictionary where keys are evaluation metrics and values are lists of the top N model names.
    """
    top_models = []

    for metric, result in results.items():
        sorted_results = sorted(result.items(), key=lambda x: mean(x[1]), reverse=not ascending)
        for rank, (name, scores) in enumerate(sorted_results[:top_n], 1):
            if name not in top_models:
                top_models.append(name)

    return top_models

def nested_cv_with_dim_reduction(X, y, model, param_grid, feature_selector, dim_reduction, scoring_metric, outer_folds=5, inner_folds=3):
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    best_params = []
    selected_features = []
    outer_scores = []

    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        

        # Apply feature selection
        features_selected = fs.select_features_with_statistic_test(X_train, y_train, feature_selector)
        X_train = X_train.loc[:, features_selected]
        X_test = X_test.loc[:, features_selected]
        selected_features.append(features_selected)

        # Create a pipeline
        pipeline = Pipeline([
            ('reduce_dim', dim_reduction),
            ('clf', model)
        ])

        # Perform GridSearch
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=scoring_metric)
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params.append(grid_search.best_params_)

        # Evaluate on the test set
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)

    return best_params, selected_features, outer_scores

def evaluate_models(X, y, models, feature_selector, metrics_list=['f1_macro', make_scorer(matthews_corrcoef, greater_is_better=True)]):
    """
    Evaluate multiple machine learning models using cross-validation.

    Args:
        X (array-like): Input data.
        y (array-like): Target labels.
        models (dict): Dictionary of machine learning models.
        param_grids (dict): Dictionary of parameter grids for each model.
        metrics_list (list): List of evaluation metrics to be used (default includes 'f1_macro' and Matthews correlation coefficient).

    Returns:
        dict: A dictionary where keys are evaluation metrics and values are dictionaries of model names and corresponding scores.
    """
    param_grids = pg.param_grids
    metric_results = {}
    for metric in metrics_list:
        results = {}
        for name, model in tqdm(models.items(), desc=f"Models Evaluation with {metric}"):
            try:
                param_grid = param_grids[name]
                scores = evaluate_model(X, y, model, param_grid, metric, feature_selector)
                results[name] = scores
            except Exception as e:
                results[name] = None
                print(f"Error evaluating model {name}: {str(e)}")
        metric_results[metric] = results
    return metric_results 

def evaluate_model(X, y, model, param_grid, scoring_metric, feature_selector_key, dim_reduction_key='pca', outer_folds=5, inner_folds=3):
    """
    Evaluate a single machine learning model using nested cross-validation.

    Args:
        X (array-like): Input data.
        y (array-like): Target labels.
        model: The machine learning model to be evaluated.
        param_grid (dict or list of dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        scoring_metric (str): The evaluation metric to be used.
        feature_selector_key (str): Key to select the feature selector to be used.
        dim_reduction_key (str): Key to select the dimensionality reduction technique to be used (default is 'pca').
        outer_folds (int): Number of folds for the outer cross-validation (default is 5).
        inner_folds (int): Number of folds for the inner cross-validation (default is 3).

    Returns:
        array-like: Array of scores obtained from the outer cross-validation.
    """
    dim_reduction = dr.dimensionality_red_dict[dim_reduction_key]
    
    best_params, selected_features, outer_scores = nested_cv_with_dim_reduction(X, y, model, param_grid, feature_selector_key, dim_reduction, scoring_metric, outer_folds=outer_folds, inner_folds=inner_folds)

    # Find the most frequently selected features
    feature_counts = Counter([feature for sublist in selected_features for feature in sublist])
    most_common_features = [feature for feature, count in feature_counts.most_common() if count > 1]

    # Find the most common best parameters
    param_counts = Counter([str(param) for param in best_params])
    most_common_params = eval(param_counts.most_common(1)[0][0])

    # Train the model with the most common features and parameters
    model.set_params(**most_common_params['clf'])
    model.fit(X[most_common_features], y)

    # Evaluate the model on the entire dataset
    score = evaluate_score(X, y, model, most_common_features, scoring_metric)

    return most_common_features, most_common_params, score

def evaluate_score(X, y, model, features, scoring_metric):
    """
    Evaluate the model performance on the entire dataset.

    Args:
        X (array-like): Input data.
        y (array-like): Target labels.
        model: The trained machine learning model.
        features (list): List of selected features.
        scoring_metric (str): The evaluation metric to be used.

    Returns:
        float: The score obtained by the model.
    """
    y_pred = model.predict(X[features])
    if scoring_metric == 'accuracy':
        return accuracy_score(y, y_pred)
    elif scoring_metric == 'f1_macro':
        return f1_score(y, y_pred, average='macro')
    elif scoring_metric == 'matthews_corrcoef':
        return matthews_corrcoef(y, y_pred)
    else:
        raise ValueError(f"Unsupported scoring metric: {scoring_metric}")
    



