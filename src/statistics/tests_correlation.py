from itertools import combinations
import pandas as pd
from scipy.stats import spearmanr

def pairwise_spearman_test(X_train, y_train, significant_features='all'):
    """
    Perform Spearman correlation test for each pair of groups and return results for each feature.  

    Args:
    - X_train: DataFrame pandas containing the data.
    - y_train: Pandas Series containing the group labels.
    - significant_features: List of feature names to perform Spearman correlation test.
    """

    if significant_features == 'all':
        significant_features = X_train.columns
        
    group_combinations = combinations(y_train.unique(), 2)
    test_results_dict = {}

    for group1, group2 in group_combinations:
        group1_data = X_train[y_train == group1]
        group2_data = X_train[y_train == group2]
        results = {'Feature': [], 'Correlation': [], 'P-value': []}
        
        for feature in significant_features:
            correlation, p_value = spearmanr(group1_data[feature], group2_data[feature])
            results['Feature'].append(feature)
            results['Correlation'].append(correlation)
            results['P-value'].append(p_value)
        
        test_results_df = pd.DataFrame(results)
        test_results_dict[(group1, group2)] = test_results_df

    return test_results_dict  