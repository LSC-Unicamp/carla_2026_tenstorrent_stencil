from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUR_DIR = BASE_DIR / "our"
TAYLOR_DIR = BASE_DIR / "taylor"
OUTPUT_PLOT = BASE_DIR / "speedup_our_vs_taylor_por_nz.pdf"
FONT_SCALE = 1.25
FILE_PATTERN = re.compile(r"nz(\d+)\.csv$")
STENCIL_SCOPE = "Stencil"
STENCIL_RISC = "TRISC_0"
LABEL_OUR = "Our's"
LABEL_TAYLOR = "Taylor et al."


def _font_size(size: float) -> float:
    return size * FONT_SCALE


def ler_speedup_por_nz() -> list[dict[str, int | float | str]]:
    arquivos_our = _mapear_arquivos_por_nz(OUR_DIR)
    arquivos_taylor = _mapear_arquivos_por_nz(TAYLOR_DIR)
    nz_comuns = sorted(set(arquivos_our) & set(arquivos_taylor))

    if not nz_comuns:
        raise ValueError("Nenhum nz em comum encontrado entre 'our' e 'taylor'")

    resultados = []

    for nz in nz_comuns:
        our_tempo_us, our_origem = _ler_tempo_relevante_us(arquivos_our[nz])
        taylor_tempo_us, taylor_origem = _ler_tempo_relevante_us(arquivos_taylor[nz])

        resultados.append(
            {
                "nz": nz,
                "tempo_our_us": our_tempo_us,
                "tempo_taylor_us": taylor_tempo_us,
                "speedup": taylor_tempo_us / our_tempo_us,
                "origem_our": our_origem,
                "origem_taylor": taylor_origem,
            }
        )

    return resultados


def plotar_speedup_por_nz(
    dados: list[dict[str, int | float | str]],
    output_path: Path = OUTPUT_PLOT,
) -> Path:
    nz_values = [int(item["nz"]) for item in dados]
    speedups = [float(item["speedup"]) for item in dados]

    plt.figure(figsize=(12, 5))
    plt.plot(nz_values, speedups, marker="o", markersize=3, linewidth=1)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.xlabel("nz", fontsize=_font_size(12))
    plt.ylabel(f"Speedup ({LABEL_TAYLOR} / {LABEL_OUR})", fontsize=_font_size(12))
    plt.title("Speedup by nz - Taylor et al. vs Our's", fontsize=_font_size(14))
    plt.xticks(fontsize=_font_size(10))
    plt.yticks(fontsize=_font_size(10))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def _mapear_arquivos_por_nz(csv_dir: Path) -> dict[int, Path]:
    arquivos_por_nz = {}

    for csv_path in sorted(csv_dir.glob("nz*.csv"), key=_chave_ordenacao_arquivo):
        nz = _extrair_nz(csv_path)
        arquivos_por_nz[nz] = csv_path

    return arquivos_por_nz


def _chave_ordenacao_arquivo(csv_path: Path) -> int:
    return _extrair_nz(csv_path)


def _extrair_nz(csv_path: Path) -> int:
    match = FILE_PATTERN.search(csv_path.name)
    if not match:
        raise ValueError(f"Nome de arquivo invalido: {csv_path.name}")
    return int(match.group(1))


def _ler_tempo_relevante_us(csv_path: Path) -> tuple[float, str]:
    freq_mhz, tempo_stencil_ciclos = _ler_janela_stencil_trisc0_core_unico(csv_path)
    if tempo_stencil_ciclos is not None:
        return tempo_stencil_ciclos / freq_mhz, "janela_stencil_trisc0_core_unico"

    freq_mhz, tempo_total_ciclos = _ler_tempo_total_csv(csv_path)
    return tempo_total_ciclos / freq_mhz, "algoritmo_total"


def _ler_janela_stencil_trisc0_core_unico(csv_path: Path) -> tuple[float, int | None]:
    with csv_path.open("r", newline="") as file:
        reader = csv.reader(file)

        primeira_linha = next(reader)
        freq_mhz = _extrair_frequencia_mhz(primeira_linha)

        next(reader)

        tempos_por_core = {}

        for row in reader:
            if (
                len(row) <= 11
                or row[10].strip() != STENCIL_SCOPE
                or row[3].strip() != STENCIL_RISC
            ):
                continue

            core = (row[1].strip(), row[2].strip())
            tipo_evento = row[11].strip()
            tempo_ciclos = int(row[5].strip())

            if tipo_evento == "ZONE_START":
                if core not in tempos_por_core:
                    tempos_por_core[core] = {
                        "primeiro_inicio": tempo_ciclos,
                        "ultimo_fim": None,
                    }
                continue

            if tipo_evento != "ZONE_END" or core not in tempos_por_core:
                continue

            tempos_por_core[core]["ultimo_fim"] = tempo_ciclos

    if not tempos_por_core:
        return freq_mhz, None

    core_escolhido = min(tempos_por_core)
    primeiro_inicio = tempos_por_core[core_escolhido]["primeiro_inicio"]
    ultimo_fim = tempos_por_core[core_escolhido]["ultimo_fim"]

    if primeiro_inicio is None or ultimo_fim is None:
        return freq_mhz, None

    return freq_mhz, ultimo_fim - primeiro_inicio


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
    dados = ler_speedup_por_nz()
    grafico = plotar_speedup_por_nz(dados)
    maior_speedup = max(dados, key=lambda item: float(item["speedup"]))
    menor_speedup = min(dados, key=lambda item: float(item["speedup"]))

    print(f"Grafico salvo em: {grafico}")
    print(f"Quantidade de nz comparados: {len(dados)}")
    print(
        f"Maior speedup: {float(maior_speedup['speedup']):.4f}x em nz={int(maior_speedup['nz'])}"
    )
    print(
        f"Menor speedup: {float(menor_speedup['speedup']):.4f}x em nz={int(menor_speedup['nz'])}"
    )
