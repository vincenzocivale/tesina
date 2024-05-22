from imblearn.over_sampling  import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd

def convert_float32(df):
    """
    Convert all float columns in a DataFrame to float32.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: DataFrame with float columns converted to float32.
    """
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def scale_numeric_features(X):
    """
    Scale numeric features in a DataFrame while excluding categorical columns.

    Args:
        X (DataFrame): The DataFrame containing the features to be scaled.
        y (Series or array): The target variable used to group the data.

    Returns:
        DataFrame: The DataFrame with scaled features.
    """
    scaler = StandardScaler()

    scaled_df = X.copy()

    numeric_columns = scaled_df.select_dtypes(include=['number']).columns.tolist()

    scaled_features = scaler.fit_transform(scaled_df[numeric_columns])

    scaled_df[numeric_columns] = scaled_features

    return scaled_df

def noramlize_data(df):
    """
    Normalize the data
    """
    df['Data'] = df['Data'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    return df

def balance_dataset(X_train, y_train):
    """
    Balance the dataset using SMOTE.

    Args:
        X_train (DataFrame or array-like): Features of the dataset.
        y_train (Series or array-like): Target variable of the dataset.

    Returns:
        X_resampled (DataFrame or array-like): Resampled features.
        y_resampled (Series or array-like): Resampled target variable.
    """
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def calculate_patient_median(X, y, patient_feature, features):
    """
    Calculate the median value of features associated with each patient.

    Args:
        X (pd.DataFrame): DataFrame containing the features.
        y (pd.Series or np.array): Series or array containing the target labels.
        patient_feature (str): The name of the column containing the patient identifier.
        features (list): List of feature names to calculate the median.

    Returns:
        pd.DataFrame: Median values of features associated with each patient.
    """
    df = pd.concat([X, y], axis=1)
    grouped_df = df.groupby(patient_feature)[features].median()
    grouped_df = grouped_df.reset_index()
    median_df = grouped_df.merge(df[[patient_feature, y.name]].drop_duplicates(), on=patient_feature)
    filtered_X = median_df.drop(columns=[y.name])
    filtered_y = median_df[y.name]
    return filtered_X, filtered_y