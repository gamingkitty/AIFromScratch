import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import numpy as np


def plot_training_csv(path, ema_span=100):
    df = pd.read_csv(path).sort_values("step").reset_index(drop=True)

    df["ema_loss"] = df["loss"].ewm(span=ema_span, adjust=False).mean()

    # Raw slope of EMA loss vs step
    df["ema_loss_rate"] = np.gradient(df["ema_loss"].to_numpy(), df["step"].to_numpy())

    fig, axes = plt.subplots(1, 1, figsize=(16, 4))

    axes.plot(df["step"], df["loss"], label="loss")
    axes.plot(df["step"], df["ema_loss"], label=f"ema_loss (span={ema_span})")
    axes.set_xlabel("step")
    axes.set_ylabel("loss")
    axes.set_title("Loss vs Step")
    axes.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_training_csv(path="Data/coco_yolo_v2", ema_span=100)