from scipy.stats import friedmanchisquare, f_oneway, mannwhitneyu
from itertools import combinations
import pandas as pd



def friedman_test(X, y, seed=42):
    """
    Esegue il test di Friedman per confrontare le differenze tra i gruppi su misure ripetute,
    gestendo il caso in cui le classi siano sbilanciate.

    :return: DataFrame contenente i risultati del test
    """
    result_df = pd.DataFrame(columns=['Feature', 'Group', 'Statistic', 'P-value'])

    for feature in X.columns:
        groups = []
        for group in y.unique():
            feature_values = X.loc[y == group, feature]
            groups.append(feature_values)
        max_group_size = max(len(group) for group in groups)
        balanced_groups = [group.sample(n=max_group_size, replace=True, random_state=seed) if len(group) < max_group_size else group for group in groups]
        f_statistic, p_value = friedmanchisquare(*balanced_groups)
        
        result_df = pd.concat([result_df, pd.DataFrame([{'Feature': feature, 'Group': group, 'Statistic': f_statistic, 'P-value': p_value}])], ignore_index=True)

    return pd.DataFrame(result_df)


def anova_test(X, y):
    """
    Perform ANOVA test to identify differences between groups for each feature.

    Returns:
    - DataFrame pandas containing the results of ANOVA test for each feature.
    """
    result_df = pd.DataFrame(columns=['Feature', 'Group', 'Statistic', 'P-value'])

    for feature in X.columns:
        groups = []
        for group in y.unique():
            feature_values = X.loc[y == group, feature]
            groups.append(feature_values)
        f_statistic, p_value = f_oneway(*groups)
        result_df = pd.concat([result_df, pd.DataFrame([{'Feature': feature, 'Group': group, 'Statistic': f_statistic, 'P-value': p_value}])], ignore_index=True)
    return result_df

def mann_whitney_test(X, y):
    """
    Perform Mann-Whitney U test to identify differences between groups for each feature.

    Returns:
    - DataFrame pandas containing the results of Mann-Whitney U test for each feature.
    """
    results_df = pd.DataFrame(columns=['Feature', 'Group', 'Statistic', 'P-value'])
    for feature in X.columns:
        groups = []
        for group in y.unique():
            feature_values = X.loc[y == group, feature]
            groups.append(feature_values)
        u_statistic, p_value = mannwhitneyu(*groups)
        results_df = pd.concat([results_df, pd.DataFrame([{'Feature': feature, 'Group': group, 'Statistic': u_statistic, 'P-value': p_value}])], ignore_index=True)

    return results_df


def pairwise_mann_whitney_test(X, y, significant_features):
    """
    Perform Mann-Whitney U test for each pair of groups and return results for each feature.

    Args:
    - X: DataFrame pandas containing the data.
    - y: Pandas Series containing the group labels.
    - features: List of feature names to perform Mann-Whitney U test.

    Returns:
    - Dictionary containing DataFrame with test results for each pair of groups.
    """
    group_combinations = combinations(y.unique(), 2)
    test_results_dict = {}

    for group1, group2 in group_combinations:
        group1_data = X[y == group1]
        group2_data = X[y == group2]
        results = {'Feature': [], 'F-statistic': [], 'P-value': []}
        
        for feature in significant_features:
            u_statistic, p_value = mannwhitneyu(group1_data[feature], group2_data[feature])
            results['Feature'].append(feature)
            results['F-statistic'].append(u_statistic)
            results['P-value'].append(p_value)
        
        test_results_df = pd.DataFrame(results)
        test_results_dict[(group1, group2)] = test_results_df

    return test_results_dict

