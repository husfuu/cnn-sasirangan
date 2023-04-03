from src.utils.constants import EVAL_DATA_DIR, HIST_MODEL_PATH
import matplotlib.pyplot as plt
import numpy as np

def load_eval_data(eval_data_dir=EVAL_DATA_DIR):
    with open(EVAL_DATA_DIR, 'rb') as f:
        train_data = np.load(f)
        train_labels = np.load(f)
        val_data = np.load(f)
        val_labels = np.load(f)

    return train_data, train_labels, val_data, val_labels

def eval_visualization():
    history = np.load(HIST_MODEL_PATH)
    fig, ax = plt.subplots(1, 3, figsize = (30, 5))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "auc", "loss"]):
        ax[i].plot(history_res.history[metric])
        ax[i].plot(history_res.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("Epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

def eval_result():
    pass
