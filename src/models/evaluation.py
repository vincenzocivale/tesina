from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from statistics import mean, stdev
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef
from tqdm import tqdm
from . import model as md
from . import param_grid as pg

# Valuta un singolo modello con cross-validazione
def evaluate_model(X, y, model, metric, cv=5):
    """
    Evaluate a single machine learning model using cross-validation.

    Args:
        X (array-like): Input data.
        y (array-like): Target labels.
        model: The machine learning model to be evaluated.
        metric: The evaluation metric to be used.
        cv (int, cross-validation generator, or an iterable): Determines the cross-validation splitting strategy (default is 5).

    Returns:
        array-like: Array of scores obtained from cross-validation.
    """
    stratified_kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    pipeline = md.make_pipeline(model)
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=stratified_kfold, n_jobs=-1, error_score="raise")
    return scores

# Valuta più modelli con cross-validazione
def evaluate_models(X, y, models, metrics_list=['f1_macro', make_scorer(matthews_corrcoef, greater_is_better=True)]):
    """
    Evaluate multiple machine learning models using cross-validation.

    Args:
        X (array-like): Input data.
        y (array-like): Target labels.
        models (dict): Dictionary of machine learning models.
        metrics_list (list): List of evaluation metrics to be used (default includes 'f1_macro' and Matthews correlation coefficient).

    Returns:
        dict: A dictionary where keys are evaluation metrics and values are dictionaries of model names and corresponding scores.
    """
    metric_results = {}
    for metric in metrics_list:
        results = {}
        for name, model in tqdm(models.items(), desc=f"Models Evaluation with {metric}"):
            try:
                scores = evaluate_model(X, y, model, metric)
                results[name] = scores
            except Exception as e:
                results[name] = None
                print(f"Error evaluating model {name}: {str(e)}")
        metric_results[metric] = results
    return metric_results

# Riassumi i risultati
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


# Valuta i modelli ottimizzati
def evaluate_optimized_models(X, y, metric_results, metric='f1_macro', cv=5):
    """
    Evaluate optimized machine learning models using cross-validation and return optimal parameters.

    Args:
        X (array-like): Input data.
        y (array-like): Target labels.
        metric_results (dict): Dictionary containing evaluation results where keys are evaluation metrics and values are dictionaries of model names and corresponding scores.
        metric (str): The evaluation metric to be used (default is 'f1_macro').
        cv (int, cross-validation generator, or an iterable): Determines the cross-validation splitting strategy (default is 5).

    Returns:
        dict: A dictionary where keys are model names and values are optimal hyperparameters.
    """

    models = md.define_models()
    model_names = find_top_models(metric_results)
    optimal_parameters = {}

    for model_name in model_names:
        model = models[model_name]
        if md.fine_tune_model(model_name, model, X, y, metric, cv) is not None:
            best_params = md.fine_tune_model(model_name, model, X, y, metric, cv)
            optimal_parameters[model_name] = best_params
            print(f"Best hyperparameters for model {model_name}: {best_params}")
            scores = evaluate_model(X, y, model.set_params(**best_params), metric, cv)
        else:
            scores = evaluate_model(X, y, model, metric, cv)
        mean_score, std_score = mean(scores), stdev(scores)
        print(f"Model {model_name} - Score={mean_score:.3f} (+/- {std_score:.3f})")
        print("---------------------------------------------------------------")
        

    return optimal_parameters
