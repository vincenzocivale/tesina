import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import logging
from typing import Optional, List, Tuple


logging.basicConfig(level=logging.INFO)


def select_n_data_per_patient(df, n, patient_feature, signal_feature):
    """
    Select n data points for each patient.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        n (int): Number of data points to select for each patient.
        patient_feature (str): The name of the column containing the patient identifier.
        signal_feature (str): The name of the column containing the signal identifier.

    Returns:
        pd.DataFrame: Selected data.
    """
    selected_data_list = []

    grouped_df = df.groupby([patient_feature, signal_feature])

    for (_, _), sub_df in grouped_df:
        selected_data = sub_df.head(n)
        selected_data_list.append(selected_data)

    selected_df = pd.concat(selected_data_list, ignore_index=True)
    return selected_df


def filter_rows_by_values(X, y, target_values):
    """
    Filtra le righe di X e y in base ai valori specificati in y.

    Args:
    - X: DataFrame pandas contenente le feature.
    - y: Serie pandas contenente le etichette di classe.
    - target_values: Lista dei valori target da mantenere.

    Returns:
    - filtered_X: DataFrame pandas contenente le righe filtrate di X.
    - filtered_y: Serie pandas contenente i valori target filtrati di y.
    """
    # Seleziona solo i valori target specificati in y
    mask = y.isin(target_values)
    filtered_X = X.loc[mask]
    filtered_y = y.loc[mask]
    
    return filtered_X, filtered_y

def filter_outliers_by_group(X: pd.DataFrame, y: pd.Series, contamination: float = 0.05, features_to_ignore: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter outliers in X for each unique value in y con IsolationForest.

    Args:
        X (pd.DataFrame): DataFrame containing the features.
        y (pd.Series or np.array): Series or array containing the target labels.
        contamination (float): Proportion of outliers in the data.
        features_to_ignore (list): List of feature names to ignore during outlier detection.

    Returns:
        pd.DataFrame: Filtered features.
        pd.Series: Filtered target labels.
    """
    if features_to_ignore is None:
        features_to_ignore = []

    filtered_X_list = []
    filtered_y_list = []

    for label in np.unique(y):
        X_label = X[y == label]
        y_label = y[y == label]

        features_to_analyze = X_label.drop(features_to_ignore, axis=1)

        iso = IsolationForest(contamination=contamination)
        yhat = iso.fit_predict(features_to_analyze)

        mask = yhat != -1
        filtered_X_label = X_label[mask]
        filtered_y_label = y_label[mask]

        removed_rows_count = len(X_label) - len(filtered_X_label)
        logging.info(f"Removed {removed_rows_count} rows for label {label}")

        filtered_X_list.append(filtered_X_label)
        filtered_y_list.append(filtered_y_label)

    filtered_X = pd.concat(filtered_X_list, ignore_index=True)
    filtered_y = pd.concat(filtered_y_list, ignore_index=True)

    return filtered_X, filtered_y

def filter_fit_value(X, y, fit_value, fit_feature):
    """
    Filter X and y by the fit value of the model.

    Args:
        X (pd.DataFrame): DataFrame containing the features.
        y (pd.Series or np.array): Series or array containing the target labels.
        fit_value (float): The value to filter the data.
        fit_feature (str): The name of the column containing the labels of fit.

    Returns:
        pd.DataFrame: Filtered features.
        pd.Series: Filtered target labels.
    """
    mask = X[fit_feature] >= fit_value
    filtered_X = X[mask]
    filtered_y = y[mask]
    return filtered_X, filtered_y
