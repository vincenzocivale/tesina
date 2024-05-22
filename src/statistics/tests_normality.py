from scipy.stats import shapiro, kstest
import pandas as pd

def shapiro_test(X, y):

    result_df = pd.DataFrame(columns=['Feature', 'Group', 'Statistic', 'P-value'])

    for feature in X.columns:
        for group in y.unique():
            # Seleziona solo i valori corrispondenti al gruppo corrente
            feature_values = X.loc[y == group, feature]

            stat, p_value = shapiro(feature_values)
            result_df = pd.concat([result_df, pd.DataFrame([{'Feature': feature, 'Group': group, 'Statistic': stat, 'P-value': p_value}])], ignore_index=True)
    
    return result_df

def kolmogorov_smirnov_test(X, y):
    """
    Perform Kolmogorov-Smirnov test for each feature and group combination.

    Returns:
    - DataFrame pandas containing the test results.
    """
    result_df = pd.DataFrame(columns=['Feature', 'Group', 'Statistic', 'P-value'])

    for feature in X.columns:
        for group in y.unique():
           # Seleziona solo i valori corrispondenti al gruppo corrente
            feature_values = X.loc[y == group, feature]
                                         
            # Kolmogorov-Smirnov test for normality
            stat, p_value = kstest(feature_values, 'norm')
            result_df = pd.concat([result_df, pd.DataFrame([{'Feature': feature, 'Group': group, 'Statistic': stat, 'P-value': p_value}])], ignore_index=True)
    
    return result_df
