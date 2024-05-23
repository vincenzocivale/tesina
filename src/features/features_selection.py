from ..statistics import tests_difference as td


def select_features_with_statistic_test(X, y, statistical_test, p_value_threshold=0.05):
    try:
        if statistical_test == 'friedman':
            test_result = td.friedman_test(X, y)
        elif statistical_test == 'anova':
            test_result = td.anova_test(X, y)
        elif statistical_test == 'mann_whitney':
            test_result = td.mann_whitney_test(X, y)
        elif statistical_test == None:
            return X.columns.tolist()
        else:
            raise ValueError("Invalid statistical test specified.")
        
        features_selected = test_result[test_result['P-value'] < p_value_threshold]['Feature'].tolist()
        
        return features_selected
    except Exception as e:
        print(f"An error occurred at feature selection: {str(e)}")
        return None




