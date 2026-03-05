"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath):

    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("File not found. Please try another filepath.")
    except ValueError:
        print("File is empty or corrupted. Please try again.")
    
    return df


def preprocess_data(df):

    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    df = df.copy()

    cols_to_drop = df.columns[df.isnull().mean() > 0.3]
    df = df.drop(cols_to_drop, axis=1)

    feature_cols = [c for c in df.columns if c != 'chol']
    cat_cols = [c for c in feature_cols if df[c].dtype in ['str', 'object', 'bool']]
    num_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in cat_cols:
        encoded = pd.get_dummies(df[col], prefix=col, dtype = int)
        df = df.drop(col, axis = 1)
        df = pd.concat([df, encoded], axis=1)

    return df


def prepare_regression_data(df, target='chol'):

    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """

    df = df.copy()
    df = df.dropna(subset=[target])
    X = df.drop(target, axis=1)
    y = df[target]

    return (X, y)


def prepare_classification_data(df, target='num'):

    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """

    df = df.copy()

    if target not in df.columns:
        target = 'num'

    df[target] = (df[target] > 0).astype(int)
    df = df.drop('chol', axis = 1)
    X = df.drop(target, axis=1)
    y = df[target]

    return (X, y)


def split_and_scale(X, y, test_size=0.2, random_state=42):

    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,      
    random_state=random_state,    
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler)

