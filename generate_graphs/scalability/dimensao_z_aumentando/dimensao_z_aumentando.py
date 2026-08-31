from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PLOT = BASE_DIR / "tempo_por_dimensao_z.pdf"
FONT_SCALE = 2.0
Z_PATTERN = re.compile(r"z-(\d+)\.csv$")


def _font_size(size: float) -> float:
    return size * FONT_SCALE


def ler_tempos_totais_dimensao_z(csv_dir: Path = DATA_DIR) -> list[dict[str, float]]:
    resultados = []

    for csv_path in sorted(csv_dir.glob("z-*.csv"), key=_extrair_dimensao_z):
        dimensao_z = _extrair_dimensao_z(csv_path)
        freq_mhz, tempo_total_ciclos = _ler_tempo_total_csv(csv_path)
        tempo_total_ms = tempo_total_ciclos / (freq_mhz * 1_000)

        resultados.append(
            {
                "dimensao_z": dimensao_z,
                "tempo_total_ciclos": tempo_total_ciclos,
                "tempo_total_ms": tempo_total_ms,
            }
        )

    return resultados


def plotar_tempo_por_dimensao_z(
    dados: list[dict[str, float]],
    output_path: Path = OUTPUT_PLOT,
) -> Path:
    dimensoes_z = [item["dimensao_z"] for item in dados]
    tempos_ms = [item["tempo_total_ms"] for item in dados]

    plt.figure(figsize=(8, 8))
    plt.plot(dimensoes_z, tempos_ms, marker="o", markersize=4, linewidth=1)
    plt.xlabel("z dimension size", fontsize=_font_size(12))
    plt.ylabel("Total runtime (ms)", fontsize=_font_size(12))
    plt.title("Total runtime vs z dimension size", fontsize=_font_size(14))
    plt.xticks(fontsize=_font_size(10))
    plt.yticks(fontsize=_font_size(10))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def _extrair_dimensao_z(csv_path: Path) -> int:
    match = Z_PATTERN.search(csv_path.name)
    if not match:
        raise ValueError(f"Nome de arquivo invalido: {csv_path.name}")
    return int(match.group(1))


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
    dados = ler_tempos_totais_dimensao_z()
    grafico = plotar_tempo_por_dimensao_z(dados)
    print(f"Grafico salvo em: {grafico}")
