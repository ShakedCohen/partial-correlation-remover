import pandas as pd
import pingouin as pg
from fairlearn.preprocessing import CorrelationRemover
from sklearn.base import BaseEstimator, TransformerMixin


"""
Advantages of inherenting from BaseEstimator and TransformerMixin:

Using the BaseEstimator and TransformerMixin classes in a custom transformer class provides several benefits:

Consistency with scikit-learn API: By inheriting from BaseEstimator and TransformerMixin, your transformer class will have a familiar scikit-learn API that can be used with other scikit-learn functions and pipelines.

Automatic hyperparameter tuning: If you define hyperparameters for your transformer, they can be automatically tuned using scikit-learn's GridSearchCV or RandomizedSearchCV functions.

Ability to combine with other transformers in a pipeline: Your transformer can be combined with other transformers in a scikit-learn pipeline to create a powerful data processing and modeling workflow.

Easier testing and maintenance: By encapsulating the transformation logic in a separate class, it becomes easier to test and maintain the code, as each component can be tested and modified independently.

"""

class PartialCorrelationRemover(BaseEstimator, TransformerMixin):
    """Transformer that removes features that are highly correlated with sensitive attributes based on partial correlation coefficients.
    Partial correlation coefficients are calculated between each sensitive attribute and each non-sensitive feature, controlling for the other non-sensitive features.
    Features with an absolute correlation coefficient greater than the given threshold will be removed from the dataset.
    
    Parameters
    ----------
    sensitive_attrs : list-like
        A list of column names for sensitive attributes.
    threshold : float, optional (default=0.1)
        The threshold value for correlation coefficients. Features with an absolute correlation coefficient greater than this value will be removed.
    
    Attributes
    ----------
    high_corr_features_ : list
        A list of column names for the features that are highly correlated with sensitive attributes and have been removed.
    
    partial_corr_ : dict
        A dictionary holding the partial correlation values, with sensitive attributes as keys, and correlation with non-sensitive features as values.
    """
    def __init__(self, sensitive_attrs, threshold=0.1):
        self.sensitive_attrs = sensitive_attrs
        self.threshold = threshold
        self.partial_corr_ = None
        self.high_corr_features_ = None
    
    def fit(self, X, y=None):
        # Extract sensitive attributes and non-sensitive features from input data
        self.sensitive_features_data = X[self.sensitive_attrs]
        self.non_sensitive_features_data = X.drop(self.sensitive_attrs, axis=1)
        
        # Calculate partial correlation coefficients between each sensitive attribute and each non-sensitive feature,
        # controlling for the other non-sensitive features
        self.partial_corr_ = {}
        for sensitive_col in self.sensitive_attrs:
            self.partial_corr_[sensitive_col] = {}
            for feature in self.non_sensitive_features_data.columns:
                covar = self.non_sensitive_features_data.drop(feature, axis=1).columns.tolist()
                corr = pg.partial_corr(data=X, x=feature, y=sensitive_col, covar=covar)["r"][0]
                self.partial_corr_[sensitive_col][feature] = corr
        
        # Identify features with high correlation to sensitive attributes
        self.high_corr_features_ = []
        for sensitive_col, partial_corr_dict in self.partial_corr_.items():
            for feature, corr_coeff in partial_corr_dict.items():
                if abs(corr_coeff) > self.threshold:
                    self.high_corr_features_.append(feature)
        return self
    
    def transform(self, X):
        # Remove high-correlated features from input data
        X_partial_cr = X.drop(self.high_corr_features_, axis=1)
        return X_partial_cr
    


class LinearTransformationCorrelationRemover():
    # fairlearn

    def __init__(self, *args, **kwargs) -> None:
        return CorrelationRemover(*args, **kwargs)
