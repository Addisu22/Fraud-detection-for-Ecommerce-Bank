import shap
import matplotlib.pyplot as plt
import os

def plot_summary(shap_values, X_sample, save_path=None):
    plt.figure()
    shap.plots.beeswarm(shap_values)
    if save_path:
        plt.savefig(save_path + "_summary.png")
    plt.show()

def plot_force(explainer, shap_values, row_idx=0, save_path=None):
    try:
        force_plot = shap.plots.force(
            explainer.expected_value,
            shap_values.values[row_idx],
            shap_values.data[row_idx],
            matplotlib=True
        )
        if save_path:
            plt.savefig(save_path + f"_force_{row_idx}.png")
        plt.show()
    except Exception as e:
        print(f"Error generating force plot: {e}")
