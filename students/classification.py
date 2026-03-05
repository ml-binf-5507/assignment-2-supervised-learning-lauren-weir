"""
Classification functions for logistic regression and k-nearest neighbors.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_grid(X_train, y_train, param_grid=None):

    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

    gs = GridSearchCV(
        LogisticRegression(max_iter= 1000, random_state=42),
        param_grid, 
        scoring='roc_auc',          
        n_jobs=-1       
    )

    gs.fit(X_train, y_train)

    return gs
    

    """
    Train logistic regression models with grid search over hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix
    y_train : np.ndarray or pd.Series
        Training target vector (binary)
    param_grid : dict, optional
        Parameter grid for GridSearchCV. 
        Default: {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty': ['l2'],
                  'solver': ['lbfgs']}
        
    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object with best model
    """

    # TODO: Implement grid search for logistic regression
    # - Create LogisticRegression with max_iter=1000
    # - Use GridSearchCV with cv=5
    # - Fit on training data
    # - Return fitted GridSearchCV object
    pass


def train_knn_grid(X_train, y_train, param_grid=None):

    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
    gs = GridSearchCV(
        KNeighborsClassifier(),
        param_grid, 
        scoring='roc_auc',          
        n_jobs=-1       
    )

    gs.fit(X_train, y_train)

    return gs
    



    """
    Train k-NN models with grid search over hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix (should be scaled)
    y_train : np.ndarray or pd.Series
        Training target vector (binary)
    param_grid : dict, optional
        Parameter grid for GridSearchCV.
        Default: {'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}
        
    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object with best model
    """
    
    
    # TODO: Implement grid search for k-NN
    # - Create KNeighborsClassifier
    # - Use GridSearchCV with cv=5
    # - Fit on training data
    # - Return fitted GridSearchCV object
    pass


def get_best_logistic_regression(X_train, y_train, X_test, y_test, param_grid=None):

    lr_best = {}
    gs = train_logistic_regression_grid(X_test, y_test)

    lr_best['model'] = gs.best_estimator_
    lr_best['best_params'] = gs.best_params_
    lr_best['test_auc'] = gs.best_score_
    lr_best['cv_results_df'] = pd.DataFrame(gs.cv_results_)

    return lr_best


    """
    Get best logistic regression model with test R² evaluation.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test target
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': best fitted LogisticRegression model
        - 'best_params': best parameters found
        - 'cv_results_df': DataFrame of all CV results
    """
    # TODO: Implement best model retrieval
    # - Use train_logistic_regression_grid
    # - Extract best model
    # - Return dictionary
    pass


def get_best_knn(X_train, y_train, X_test, y_test, param_grid=None):

    knn_best = {}

    gs = train_knn_grid(X_test, y_test)

    knn_best['model'] = gs.best_estimator_
    knn_best['best_params'] = gs.best_params_
    knn_best['best_k'] = gs.best_params_['n_neighbors']
    knn_best['test_auc'] = gs.best_score_
    knn_best['cv_results_df'] = pd.DataFrame(gs.cv_results_)

    return knn_best


    """
    Get best k-NN model with test R² evaluation.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features (scaled)
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features (scaled)
    y_test : np.ndarray or pd.Series
        Test target
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': best fitted KNeighborsClassifier model
        - 'best_params': best parameters found
        - 'best_k': best n_neighbors value
        - 'cv_results_df': DataFrame of all CV results
    """
    # TODO: Implement best model retrieval
    # - Use train_knn_grid
    # - Extract best model and best_k
    # - Return dictionary
    pass
