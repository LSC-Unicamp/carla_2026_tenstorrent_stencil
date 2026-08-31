from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PLOT = BASE_DIR / "heatmap_tempo_dimensoes_x_y.pdf"
FONT_SCALE = 2.0
XY_PATTERN = re.compile(r"x(\d+)-y(\d+)\.csv$")


def _font_size(size: float) -> float:
    return size * FONT_SCALE


def ler_tempos_totais_dimensoes_xy(csv_dir: Path = DATA_DIR) -> list[dict[str, float]]:
    resultados = []

    for csv_path in sorted(csv_dir.glob("x*-y*.csv"), key=_extrair_dimensoes_xy):
        dimensao_x, dimensao_y = _extrair_dimensoes_xy(csv_path)
        freq_mhz, tempo_total_ciclos = _ler_tempo_total_csv(csv_path)
        tempo_total_ms = tempo_total_ciclos / (freq_mhz * 1000)

        resultados.append(
            {
                "dimensao_x": dimensao_x,
                "dimensao_y": dimensao_y,
                "tempo_total_ciclos": tempo_total_ciclos,
                "tempo_total_ms": tempo_total_ms,
            }
        )

    return resultados


def plotar_heatmap_tempo_dimensoes_xy(
    dados: list[dict[str, float]],
    output_path: Path = OUTPUT_PLOT,
) -> Path:
    dimensoes_x = sorted({int(item["dimensao_x"]) for item in dados})
    dimensoes_y = sorted({int(item["dimensao_y"]) for item in dados})
    indice_x = {valor: idx for idx, valor in enumerate(dimensoes_x)}
    indice_y = {valor: idx for idx, valor in enumerate(dimensoes_y)}

    matriz_tempos = np.full((len(dimensoes_y), len(dimensoes_x)), np.nan, dtype=float)

    for item in dados:
        x = int(item["dimensao_x"])
        y = int(item["dimensao_y"])
        matriz_tempos[indice_y[y], indice_x[x]] = item["tempo_total_ms"]

    min_item = min(dados, key=lambda item: item["tempo_total_ms"])
    max_item = max(dados, key=lambda item: item["tempo_total_ms"])

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matriz_tempos, origin="lower", aspect="auto", cmap="viridis")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Total runtime (ms)", fontsize=_font_size(12))
    colorbar.ax.tick_params(labelsize=_font_size(10))

    ax.set_xticks(range(len(dimensoes_x)))
    ax.set_xticklabels(dimensoes_x, fontsize=_font_size(10))
    ax.set_yticks(range(len(dimensoes_y)))
    ax.set_yticklabels(dimensoes_y, fontsize=_font_size(10))
    ax.set_xlabel("x dimension", fontsize=_font_size(12))
    ax.set_ylabel("y dimension", fontsize=_font_size(12))
    ax.set_title(
        "Heatmap of total runtime\nby x and y dimensions", fontsize=_font_size(14)
    )

    min_x_idx = indice_x[int(min_item["dimensao_x"])]
    min_y_idx = indice_y[int(min_item["dimensao_y"])]
    max_x_idx = indice_x[int(max_item["dimensao_x"])]
    max_y_idx = indice_y[int(max_item["dimensao_y"])]

    ax.scatter(
        min_x_idx, min_y_idx, s=120, facecolors="none", edgecolors="white", linewidths=2
    )
    ax.scatter(
        max_x_idx, max_y_idx, s=120, facecolors="none", edgecolors="red", linewidths=2
    )

    ax.text(
        min_x_idx + 0.15,
        min_y_idx + 0.15,
        f"min: {min_item['tempo_total_ms']:.3f} ms",
        color="white",
        fontsize=_font_size(9),
        weight="bold",
    )
    ax.text(
        max_x_idx - 0.15,
        max_y_idx + 0.15,
        f"max: {max_item['tempo_total_ms']:.3f} ms",
        color="red",
        fontsize=_font_size(9),
        weight="bold",
        ha="right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def _extrair_dimensoes_xy(csv_path: Path) -> tuple[int, int]:
    match = XY_PATTERN.search(csv_path.name)
    if not match:
        raise ValueError(f"Nome de arquivo invalido: {csv_path.name}")
    return int(match.group(1)), int(match.group(2))


def _ler_tempo_total_csv(csv_path: Path) -> tuple[float, int]:
    with csv_path.open("r", newline="") as file:
        reader = csv.reader(file)

        primeira_linha = next(reader)
        freq_mhz = _extrair_frequencia_mhz(primeira_linha)

        next(reader)

        tempos_inicio = {}
        maior_duracao_fw = None

        for row in reader:
            if len(row) <= 11:
                continue

            key = (row[1].strip(), row[2].strip(), row[3].strip(), row[10].strip())
            tipo_evento = row[11].strip()
            tempo_ciclos = int(row[5].strip())

            if tipo_evento == "ZONE_START":
                tempos_inicio[key] = tempo_ciclos
                continue

            if (
                tipo_evento != "ZONE_END"
                or key not in tempos_inicio
                or not key[3].endswith("-FW")
            ):
                continue

            duracao_fw = tempo_ciclos - tempos_inicio[key]
            maior_duracao_fw = (
                duracao_fw
                if maior_duracao_fw is None
                else max(maior_duracao_fw, duracao_fw)
            )

    if maior_duracao_fw is None:
        raise ValueError(f"Arquivo sem dados de tempo: {csv_path}")

    return freq_mhz, maior_duracao_fw


def _extrair_frequencia_mhz(primeira_linha: list[str]) -> float:
    for campo in primeira_linha:
        if "CHIP_FREQ[MHz]" in campo:
            return float(campo.split(":", maxsplit=1)[1].strip())

    raise ValueError("Frequencia do chip nao encontrada no cabecalho do CSV")


if __name__ == "__main__":
    dados = ler_tempos_totais_dimensoes_xy()
    grafico = plotar_heatmap_tempo_dimensoes_xy(dados)
    print(f"Grafico salvo em: {grafico}")
