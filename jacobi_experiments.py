import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time


import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
exec_path = str(BASE_DIR / "run_script.sh")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Export it before running jacobi_experiments.py."
        )
    return value


TT_METAL_HOME = Path(require_env("TT_METAL_HOME"))

# MODES:
# - python
# - tenstorrent
MODE = "tenstorrent"

DEFAULT_MAX_IT = 4
DEFAULT_ENGINE = ""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    print("matplotlib not installed. Not generating any graphs...")
    HAS_MATPLOTLIB = False


max_it = DEFAULT_MAX_IT
engine = DEFAULT_ENGINE


def save_to_file(volume, filename):
    volume.detach().cpu().to(torch.float32).numpy().tofile(filename)


def read_from_file(filename, shape):
    data = np.fromfile(filename, dtype=np.float32).reshape(shape)
    return torch.from_numpy(data).to(torch.float32)


def plot_slice(volume, title, z=0):
    if not HAS_MATPLOTLIB:
        return

    if z is None:
        z = volume.shape[0] // 2
    image = volume[z].to(torch.float32).cpu().numpy()

    plt.figure()
    plt.imshow(image, origin="upper", interpolation="nearest")
    plt.title(f"{title} (z={z})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()


def jacobi_python_3d(p0, max_it=max_it):
    p_curr = p0.clone()
    p_next = p0.clone()
    one_sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32)

    for it in range(max_it):
        logger.info("iteracao: %s", it)
        padded = torch.nn.functional.pad(
            p_curr, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0
        )
        p_next[:, :, :] = one_sixth * (
            padded[1:-1, 1:-1, 0:-2]
            + padded[1:-1, 1:-1, 2:]
            + padded[1:-1, 0:-2, 1:-1]
            + padded[1:-1, 2:, 1:-1]
            + padded[0:-2, 1:-1, 1:-1]
            + padded[2:, 1:-1, 1:-1]
        )
        p_curr, p_next = p_next, p_curr

    return p_curr


def jacobi(p0, mode=MODE, max_it=max_it, device=0):
    if not isinstance(p0, torch.Tensor):
        p0 = torch.tensor(p0, dtype=torch.float32)
    elif p0.dtype != torch.float32:
        p0 = p0.to(torch.float32)

    if p0.ndim != 3:
        raise ValueError(f"Expected a 3D tensor [z, y, x], got shape {tuple(p0.shape)}")

    if mode == "tenstorrent":
        logger.info("running tenstorrent")
        nz, ny, nx = p0.shape
        cmd = [
            "bash",
            exec_path,
            f"{device}",
            f"{max_it}",
            f"{ny}",
            f"{nx}",
            f"{nz}",
            engine,
        ]

        subprocess.run(cmd, check=True)

        result = read_from_file(BASE_DIR / "output.bin", p0.shape)
        plot_slice(result, f"Output (mode: {mode})")

        return result
    if mode == "python":
        result = jacobi_python_3d(p0)
        plot_slice(result, f"Output (mode: {mode})")
        return result

    raise ValueError(f"UNKNOWN MODE: {mode}")


def criar_cubo(tamanho_cubo_x, tamanho_cubo_y, tamanho_cubo_z, nx, ny, nz):
    p0 = torch.zeros((nz, ny, nx), dtype=torch.float32)
    centro_z, centro_y, centro_x = nz // 2, ny // 2, nx // 2

    inicio_z = centro_z - (tamanho_cubo_z // 2)
    fim_z = centro_z + (tamanho_cubo_z // 2)
    inicio_y = centro_y - (tamanho_cubo_y // 2)
    fim_y = centro_y + (tamanho_cubo_y // 2)
    inicio_x = centro_x - (tamanho_cubo_x // 2)
    fim_x = centro_x + (tamanho_cubo_x // 2)

    p0[inicio_z:fim_z, inicio_y:fim_y, inicio_x:fim_x] = 1.0
    return p0


def error():
    nx = 160
    ny = 160
    nz = 32
    output_path = Path.home().joinpath("error_metrics.txt")

    with output_path.open("w", encoding="ascii") as output_file:
        output_file.write("iter erro_relativo_max rmse\n")

        for i in range(1, 500):
            # p0 = criar_cubo(100, 100, 20, nx, ny, nz)
            p0 = torch.rand((nz, ny, nx), dtype=torch.float32)
            save_to_file(p0, BASE_DIR / "input.bin")
            tenstorrent_result = jacobi(p0, max_it=i)
            logger.info("%s", p0[nz // 2, ny // 2])
            plot_slice(p0, "Input")
            torch_result = jacobi_python_3d(p0, max_it=i)

            plot_slice(torch_result, "Output (mode: torch)")

            iguais = torch.allclose(
                tenstorrent_result, torch_result, rtol=1e-02, atol=1e-03
            )
            diff = tenstorrent_result - torch_result
            diff_l2 = torch.norm(diff) / torch.norm(torch_result)
            denom = torch.clamp(torch.abs(torch_result), min=1e-12)
            erro_relativo_max = torch.max(torch.abs(diff) / denom)
            rmse = torch.sqrt(torch.mean(diff * diff))

            output_file.write(f"{i} {erro_relativo_max.item():.8e} {rmse.item():.8e}\n")

            if iguais:
                logger.info("✅ O Tenstorrent e igual ao golden model em torch")
            else:
                logger.warn(
                    "❌: O resultado do Tenstorrent e diferente do golden model em torch"
                )
                logger.warn("Erro Relativo L2: %.4f", diff_l2.item())

    logger.info("Metricas salvas em %s", output_path)


def dimensao_z_aumentando(device=0, output_path=["dimensao_z_aumentando", "data"]):
    nx = 160
    ny = 160
    i = 500

    for nz in range(3, 50):
        # p0 = criar_cubo(100, 100, 20, nx, ny, nz)
        p0 = torch.rand((nz, ny, nx), dtype=torch.float32)
        save_to_file(p0, BASE_DIR / "input.bin")
        try:
            jacobi(p0, max_it=i, device=device)
        except:
            continue
        logger.info("%s", p0[nz // 2, ny // 2])
        plot_slice(p0, "Input")

        # Copiando o arquivo em $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv para $HOME/dram_vs_systolic/dimensao_z_aumentando/z-{i}.csv
        Path.home().joinpath(output_path[0], output_path[1]).mkdir(
            parents=True, exist_ok=True
        )
        shutil.copyfile(
            TT_METAL_HOME
            / "generated"
            / "profiler"
            / ".logs"
            / "profile_log_device.csv",
            Path.home().joinpath(output_path[0], output_path[1], f"z-{nz}.csv"),
        )

    logger.info("Metricas salvas em %s", Path.home().joinpath(output_path[0], output_path[1]))


def dimensoes_x_y_aumentando(device=0, output_path=["dimensao_x_y_aumentando", "data"]):
    nz = 32
    i = 500

    for nx in range(32, 32 * 12, 32):
        for ny in range(32, 32 * 12, 32):
            # p0 = criar_cubo(100, 100, 20, nx, ny, nz)
            p0 = torch.rand((nz, ny, nx), dtype=torch.float32)
            save_to_file(p0, BASE_DIR / "input.bin")
            try:
                jacobi(p0, max_it=i, device=device)
            except:
                continue
            logger.info("%s", p0[nz // 2, ny // 2])
            plot_slice(p0, "Input")

            # Copiando o arquivo em $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv para $HOME/dram_vs_systolic/dimensao_z_aumentando/x{nx}-y{ny}.csv
            Path.home().joinpath(output_path[0], output_path[1]).mkdir(
                parents=True, exist_ok=True
            )
            shutil.copyfile(
                TT_METAL_HOME
                / "generated"
                / "profiler"
                / ".logs"
                / "profile_log_device.csv",
                Path.home().joinpath(
                    output_path[0], output_path[1], f"x{nx}-y{ny}.csv"
                ),
            )
    logger.info("Metricas salvas em %s", Path.home().joinpath(output_path[0], output_path[1]))


def vs_maya_z():
    nx = 160
    ny = 160
    max_it = 1

    # Meu
    for nz in range(1, 50):
        # p0 = criar_cubo(100, 100, 20, nx, ny, nz)
        p0 = torch.rand((nz, ny, nx), dtype=torch.float32)
        save_to_file(p0, BASE_DIR / "input.bin")
        try:
            jacobi(p0, max_it=max_it)
        except:
            break
        logger.info("%s", p0[nz // 2, ny // 2])
        plot_slice(p0, "Input")

        # Copiando o arquivo em $TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv para $HOME/meu_vs_maya_z/meu/nz{nz}.csv
        Path.home().joinpath("meu_vs_maya_z", "meu").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            TT_METAL_HOME
            / "generated"
            / "profiler"
            / ".logs"
            / "profile_log_device.csv",
            Path.home().joinpath("meu_vs_maya_z", "meu", f"nz{nz}.csv"),
        )

    logger.info("Metricas salvas em %s", Path.home().joinpath("meu_vs_maya_z", "meu"))


def placa_1_vs_placa_2():
    dimensao_z_aumentando(device=0, output_path=["comparacao_placa_z", "data0"])
    dimensao_z_aumentando(device=1, output_path=["comparacao_placa_z", "data1"])
    dimensoes_x_y_aumentando(device=0, output_path=["comparacao_placa_x_y", "data0"])
    dimensoes_x_y_aumentando(device=1, output_path=["comparacao_placa_x_y", "data1"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Executa os experimentos de Jacobi com flags individuais para cada teste."
    )
    parser.add_argument(
        "--max-it",
        type=int,
        default=DEFAULT_MAX_IT,
        help="Numero de iteracoes usado por jacobi().",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="Engine passada para o fluxo Tenstorrent.",
    )
    parser.add_argument("--error", action="store_true", help="Roda error().")
    parser.add_argument(
        "--z-dimension", action="store_true", help="Roda dimensao_z_aumentando()."
    )
    parser.add_argument(
        "--xy-dimension", action="store_true", help="Roda dimensoes_x_y_aumentando()."
    )
    parser.add_argument("--vs-maya-z", action="store_true", help="Roda vs_maya_z().")
    parser.add_argument(
        "--placa-1-vs-placa-2", action="store_true", help="Roda placa_1_vs_placa_2()."
    )
    return parser.parse_args(argv)


def main(argv=None):
    global max_it, engine

    args = parse_args(argv)
    max_it = args.max_it
    engine = args.engine.strip().upper()

    logger.info("Mode: %s, max_iterations: %s", MODE, max_it)

    ran_any = False

    if args.error:
        error()
        ran_any = True
    if args.z_dimension:
        dimensao_z_aumentando()
        ran_any = True
    if args.xy_dimension:
        dimensoes_x_y_aumentando()
        ran_any = True
    if args.vs_maya_z:
        vs_maya_z()
        ran_any = True
    if args.placa_1_vs_placa_2:
        placa_1_vs_placa_2()
        ran_any = True

    if not ran_any:
        placa_1_vs_placa_2()

    if HAS_MATPLOTLIB:
        plt.show()


if __name__ == "__main__":
    main()
