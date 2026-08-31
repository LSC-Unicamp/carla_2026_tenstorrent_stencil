from pathlib import Path

import matplotlib.pyplot as plt


FONT_SCALE = 2


def _font_size(size: float) -> float:
    return size * FONT_SCALE


def plot_error_metrics(metrics_path=None, output_path=None):
    metrics_file = (
        Path(metrics_path)
        if metrics_path
        else Path(__file__).with_name("error_metrics.txt")
    )
    output_file = (
        Path(output_path)
        if output_path
        else Path(__file__).with_name("error_metrics.pdf")
    )

    iterations = []
    max_relative_errors = []
    rmse_values = []

    with metrics_file.open("r", encoding="ascii") as file:
        next(file)
        for line in file:
            iteration, max_relative_error, rmse = line.split()
            iterations.append(int(iteration))
            max_relative_errors.append(float(max_relative_error))
            rmse_values.append(float(rmse))

    fig, ax = plt.subplots(figsize=(18, 6))
    bar_width = 0.4
    left_positions = [iteration - bar_width / 2 for iteration in iterations]
    right_positions = [iteration + bar_width / 2 for iteration in iterations]

    ax.bar(
        left_positions,
        max_relative_errors,
        width=bar_width,
        label="Maximum relative error",
    )
    ax.bar(right_positions, rmse_values, width=bar_width, label="RMSE")

    ax.set_title(
        "Maximum relative error and root mean square error (RMSE) per iteration",
        fontsize=_font_size(14),
    )
    ax.set_xlabel("Iteration", fontsize=_font_size(12))
    ax.set_ylabel("Error", fontsize=_font_size(12))
    ax.legend(fontsize=_font_size(10))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(0.5, len(iterations) + 0.5)

    tick_step = max(1, len(iterations) // 20)
    ax.set_xticks(iterations[::tick_step])
    ax.tick_params(axis="x", rotation=45, labelsize=_font_size(10))
    ax.tick_params(axis="y", labelsize=_font_size(10))

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    return output_file


if __name__ == "__main__":
    generated_file = plot_error_metrics()
    print(f"Grafico salvo em: {generated_file}")
