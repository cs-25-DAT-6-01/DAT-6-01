import os
import numbers
import matplotlib.pyplot as plt


def plot_metrics(
    metrics: dict[str, list | tuple],
    run_tag: str = "run",
    out_dir: str = "plots",
    show: bool = False,
):
    if not metrics:
        raise ValueError("`metrics` dict is empty.")

    lengths = {len(v) for v in metrics.values()}
    if len(lengths) != 1:
        raise ValueError("All metric value sequences must have the same length.")
    for name, values in metrics.items():
        if not all(isinstance(x, numbers.Number) for x in values):
            raise TypeError(f"Metric '{name}' contains non‑numeric values.")

    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, next(iter(lengths)) + 1)

    for metric_name, values in metrics.items():
        plt.figure()
        plt.plot(epochs, values, marker="o")
        plt.title(f"{metric_name.capitalize()} – {run_tag}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True)

        fname = os.path.join(out_dir, f"{run_tag}_{metric_name}.png")
        plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
