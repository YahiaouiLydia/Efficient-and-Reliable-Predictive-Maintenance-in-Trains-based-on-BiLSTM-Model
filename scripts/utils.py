import matplotlib.pyplot as plt
import numpy as np
import shap

def plot_predictions(y_true, y_pred, model_name="Model"):
    """Plot a scatter of predictions vs true values."""
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='red', label='Predictions')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Reference line')
    plt.title(f'{model_name}: Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_training_history(history, model_name="Model"):
    """Plot the training and validation loss evolution."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name}: Training Loss Evolution')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_shap_summary(model, X, feature_names, model_name="Model", save_path=None):
    """Plot a SHAP summary for the model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {model_name}')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"SHAP plot saved to {save_path}")
    plt.show()