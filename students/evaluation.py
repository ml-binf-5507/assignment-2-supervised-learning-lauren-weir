"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, auc as compute_auc, r2_score
)


def calculate_r2_score(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)

    return r2
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    # TODO: Implement R² calculation
    # Use sklearn's r2_score
    pass


def calculate_classification_metrics(y_true, y_pred):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)

    return metrics



    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    
    
    # TODO: Implement metrics calculation
    # Return dictionary with all four metrics
    pass


def calculate_auroc_score(y_true, y_pred_proba):


    auroc = roc_auc_score(y_true, y_pred_proba)
    return auroc

    
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    # TODO: Implement AUROC calculation
    # Use sklearn's roc_auc_score
    pass


def calculate_auprc_score(y_true, y_pred_proba):

    auprc = average_precision_score(y_true, y_pred_proba)
    return auprc

    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    # TODO: Implement AUPRC calculation
    # Use sklearn's average_precision_score
    pass


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = calculate_auroc_score(y_true, y_pred_proba)

    
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)

        return fig
    else:
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.3f})')
        return ax


    if output_path is not None:
        plt.savefig(output_path)


    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement ROC curve plotting
    # - Calculate ROC curve using roc_curve()
    # - Calculate AUROC using auc()
    # - Plot curve with label showing AUROC score
    # - Add diagonal reference line
    # - Set labels: "False Positive Rate", "True Positive Rate"
    # - Save to output_path if provided
    # - Return figure and/or axes
    pass


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = calculate_auprc_score(y_true, y_pred_proba)

    
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP={ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)

        return fig 
    else:
        ax.plot(recall, precision, label=f'PR Curve (AP={ap:.3f})')
        return ax


    if output_path is not None:
        plt.savefig(output_path)

    

    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement PR curve plotting
    # - Calculate precision-recall curve using precision_recall_curve()
    # - Calculate AUPRC using average_precision_score()
    # - Plot curve with label showing AUPRC score
    # - Add horizontal baseline (prevalence)
    # - Set labels: "Recall", "Precision"
    # - Save to output_path if provided
    # - Return figure and/or axes
    pass


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('AUROC Curve Comparison')
    ax[0].grid(alpha=0.3)

    generate_auroc_curve(y_true, y_pred_proba_log, model_name="Logistic Regression", 
                         ax=ax[0])
    
    generate_auroc_curve(y_true, y_pred_proba_knn, model_name="k-NN", 
                        ax=ax[0])
    
    ax[0].legend(loc = 'lower right')

    
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve Comparison')
    ax[1].grid(alpha=0.3)

    generate_auprc_curve(y_true, y_pred_proba_log, model_name="Logistic Regression", 
                        ax=ax[1])
    
    generate_auprc_curve(y_true, y_pred_proba_knn, model_name="k-NN", 
                        ax=ax[1])
    
    ax[1].legend(loc = 'lower left')


    if output_path is not None:
        plt.savefig(output_path)

    return fig


    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    # TODO: Implement comparison plotting
    # - Create figure with 1x2 subplots
    # - Left: ROC curves for both models
    # - Right: PR curves for both models
    # - Add legends with AUROC/AUPRC scores
    # - Save to output_path if provided
    # - Return figure
    pass
