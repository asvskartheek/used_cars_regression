def analyze_categorical_features(train_data, categorical_features, target_variable='log_price', num_quantiles=10):
    """
    Analyze the usefulness of categorical features for predicting a target variable.

    This function performs three types of analysis:
    1. ANOVA F-test: Measures the variance between groups compared to the variance within groups.
    2. Mutual Information: Measures the mutual dependence between the feature and the target variable.
    3. Cramer's V: Measures the strength of association between the feature and the target variable.

    Parameters:
    -----------
    train_data : pandas.DataFrame
        The training dataset containing both features and target variable.
    categorical_features : list
        List of column names of categorical features to analyze.
    target_variable : str, optional (default='log_price')
        The name of the target variable column.
    num_quantiles : int, optional (default=10)
        The number of quantiles to use when discretizing the target variable for Cramer's V calculation.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the analysis results for each categorical feature.

    Notes:
    ------
    - ANOVA F-statistic: Higher values indicate a stronger relationship. 
      The p-value should be < 0.05 for statistical significance.
    - Mutual Information: Higher values indicate a stronger relationship.
    - Cramer's V: Ranges from 0 to 1. Values closer to 1 indicate a stronger association.
    """
    from scipy.stats import f_oneway
    from sklearn.feature_selection import mutual_info_regression
    import numpy as np
    from scipy.stats import chi2_contingency
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    def cramers_v(x, y):
        """
        Calculate Cramer's V statistic for categorical-categorical association.
        
        Parameters:
        -----------
        x : array-like
            A categorical variable.
        y : array-like
            Another categorical variable.
        
        Returns:
        --------
        float
            The Cramer's V statistic.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    results = {}
    le = LabelEncoder()

    for cat_feature in categorical_features:
        print(f"Analyzing {cat_feature}")
        # Temporary label encoding
        # Force all data to strings to avoid mixed type problems with categorical data
        temp_encoded = le.fit_transform(train_data[cat_feature].astype(str))
        
        # ANOVA F-statistic
        groups = [train_data[train_data[cat_feature] == cls][target_variable].values for cls in le.classes_]
        groups = [group for group in groups if len(group) > 0]  # Remove empty groups
        if len(groups) > 1:
            f_statistic, p_value = f_oneway(*groups)
        else:
            f_statistic, p_value = np.nan, np.nan
        
        # Mutual Information
        mi_score = mutual_info_regression(temp_encoded.reshape(-1, 1), train_data[target_variable])[0]
        
        # Cramer's V
        cramer_v = cramers_v(temp_encoded, pd.qcut(train_data[target_variable], q=num_quantiles, duplicates='drop'))
        
        results[cat_feature] = {
            'ANOVA F-statistic': f_statistic,
            'ANOVA p-value': round(p_value, 3),
            'Mutual Information': mi_score,
            "Cramer's V": cramer_v
        }

    return pd.DataFrame(results).T

# # Example usage:
# results_df = analyze_categorical_features(train, CATS)
# display(results_df)
