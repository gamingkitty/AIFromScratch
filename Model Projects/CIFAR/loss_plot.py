import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import numpy as np


def lr_percent_cosine_step(step, total_steps=1563*40, warmup_steps=1000, min_percent=0.05):
    if total_steps <= 1:
        return 1.0

    step = max(0, min(int(step), total_steps - 1))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps

    denom = total_steps - warmup_steps
    if denom <= 1:
        return 1.0

    t = (step - warmup_steps) / denom
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_percent + (1.0 - min_percent) * cosine


def plot_training_csv(path, ema_span=100, rate_ema_span=200):
    df = pd.read_csv(path).sort_values("step").reset_index(drop=True)

    df["ema_loss"] = df["loss"].ewm(span=ema_span, adjust=False).mean()

    # Raw slope of EMA loss vs step
    df["ema_loss_rate"] = np.gradient(df["ema_loss"].to_numpy(), df["step"].to_numpy())

    print(f"Mean loss rate in past 2000 steps: {df['ema_loss_rate'][-2000:-1].mean()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    learning_rates = [lr_percent_cosine_step(s) * 0.0005 for s in df["step"]]

    axes[0].plot(df["step"], df["loss"], label="loss")
    axes[0].plot(df["step"], df["ema_loss"], label=f"ema_loss (span={ema_span})")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss vs Step")
    axes[0].legend()

    axes[1].plot(df["step"], df["accuracy"], label="accuracy")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy vs Step")
    axes[1].legend()

    axes[2].plot(df["step"], learning_rates, label="learning_rate")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("learning_rate")
    axes[2].set_title("Learning Rate vs Step")
    axes[2].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_training_csv("Data/cifar_convolution_7", ema_span=100)
