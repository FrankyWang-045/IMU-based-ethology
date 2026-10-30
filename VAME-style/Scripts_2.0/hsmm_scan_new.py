"""测试 recording 数量增加时，BIC 最佳 K 的变化。

用法示例：
    python test_bic_vs_recordings.py
    python test_bic_vs_recordings.py --exp exp_001
    python test_bic_vs_recordings.py --exp 1 --seg-per-rec 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 兼容放在项目根目录或 Scripts_2.0 目录下
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "Scripts_2.0").is_dir():
    sys.path.insert(0, str(SCRIPT_DIR / "Scripts_2.0"))
elif (SCRIPT_DIR.parent / "Scripts_2.0").is_dir():
    sys.path.insert(0, str(SCRIPT_DIR.parent / "Scripts_2.0"))

from hsmm_scan import fit_hsmm_segments, hsmm_log_likelihood, count_params
from utils import load_config, output_dir, latest_experiment_dir


def sample_segments_per_recording(Xs, segment_length, n_per_rec, seed):
    """从每个可用 recording 中截取 n_per_rec 段。

    当 recording 足够长时，尽量无放回采样起始点以减少重叠；
    长度不足时回退到有放回采样。
    """
    rng = np.random.default_rng(seed)
    segments = []
    for X in Xs:
        if len(X) < segment_length:
            continue
        max_start = len(X) - segment_length
        n_starts = max_start + 1
        if n_starts >= n_per_rec:
            starts = rng.choice(n_starts, size=n_per_rec, replace=False)
        else:
            starts = rng.integers(0, n_starts, size=n_per_rec)
        for s in starts:
            segments.append(X[s : s + segment_length])
    if not segments:
        raise ValueError("没有可用的 recording 片段")
    return segments


def scan_best_k_for_n(Xs, k_list, d_max, segment_length, n_per_rec, seed, n_iter, n_init):
    """对当前 recording 集合做一次完整 BIC 扫描，返回所有 K 的结果。"""
    segments = sample_segments_per_recording(Xs, segment_length, n_per_rec, seed=seed)
    T_total = sum(len(seg) for seg in segments)

    results = []
    for k in k_list:
        hsmm = fit_hsmm_segments(
            segments, k, d_max, seed=seed + k, n_iter=n_iter, n_init=n_init
        )
        loglik = sum(hsmm_log_likelihood(hsmm, seg) for seg in segments)
        p = count_params(k, segments[0].shape[1])
        bic = -2 * loglik + p * np.log(T_total)
        results.append((k, loglik, p, bic))
        print(f"  K={k}: loglik={loglik:.1f}  p={p}  BIC={bic:.1f}")
    return results


def main():
    cfg = load_config()
    sc = cfg["hmm_scan"]
    hc = cfg["hmm"]

    parser = argparse.ArgumentParser(description="BIC 最佳 K 随 recording 数量变化测试")
    parser.add_argument(
        "--exp", type=str, default=None,
        help="实验目录名，如 exp_001 或 001；默认使用最新实验目录"
    )
    parser.add_argument(
        "--seg-per-rec", type=int, default=None,
        help="每个 recording 截取的段数；默认读取 config 的 hmm_scan.hsmm_scan_segments_per_recording，否则为 5"
    )
    args = parser.parse_args()

    # 1. 确定实验目录
    if args.exp:
        exp_name = args.exp if str(args.exp).startswith("exp_") else f"exp_{int(args.exp):03d}"
        exp_dir = output_dir(cfg) / exp_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
    else:
        exp_dir = latest_experiment_dir(cfg)

    print(f"使用实验目录: {exp_dir}")

    # 2. 读取 latent 文件
    latent_files = sorted(exp_dir.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError(f"{exp_dir} 中没有 *_latent.npz 文件")

    Xs_all = []
    names = []
    for p in latent_files:
        z = np.load(p)["downstream"]
        z = z[np.isfinite(z).all(axis=1)]
        if len(z) > 0:
            Xs_all.append(z)
            names.append(p.stem.replace("_latent", ""))

    # 3. 从 config 读取所有参数
    segment_length = sc.get("hsmm_scan_segment_length", 500)
    n_iter = sc.get("hsmm_scan_n_iter", 5)
    base_seed = sc.get("hsmm_scan_seed", 0)
    n_init = sc.get("hsmm_scan_n_init", 5)
    k_list = sc["k_list"]
    n_per_rec = args.seg_per_rec or sc.get("hsmm_scan_segments_per_recording", 5)

    fs = cfg["preprocess"].get("target_fs", 100.0)
    d_max = int(hc.get("hsmm_d_max_sec", 5.0) * fs)

    # 过滤长度不足以采样的 recording
    valid_idx = [i for i, X in enumerate(Xs_all) if len(X) >= segment_length]
    Xs_all = [Xs_all[i] for i in valid_idx]
    names = [names[i] for i in valid_idx]

    n_total = len(Xs_all)
    if n_total == 0:
        raise ValueError(f"没有长度 >= {segment_length} 的 latent 文件")

    print(f"可用 recordings: {n_total}, 每 recording {n_per_rec} 段, "
          f"segment_length={segment_length}, k_list={k_list}")

    # 4. 逐步增加 recording 数量，每个 N 都做一次 BIC 扫描
    summary = []          # (N, best_k, min_bic)
    full_results = {}     # N -> list of (k, loglik, p, bic)

    for N in range(1, n_total + 1):
        print(f"\n===== N={N} / {n_total}: {', '.join(names[:N])} =====")
        results = scan_best_k_for_n(
            Xs_all[:N], k_list, d_max, segment_length, n_per_rec,
            seed=base_seed + N, n_iter=n_iter, n_init=n_init,
        )
        full_results[N] = results
        bics = [bic for _, _, _, bic in results]
        best_idx = int(np.argmin(bics))
        best_k, _, _, best_bic = results[best_idx]
        summary.append((N, best_k, best_bic))
        print(f"N={N} -> best K={best_k}, BIC={best_bic:.1f}")

    # 5. 保存结果
    out_dir = exp_dir / "hsmm_scan_bic_by_n"
    out_dir.mkdir(exist_ok=True)

    summary_arr = np.array(summary, dtype=float)
    np.savetxt(
        out_dir / "bic_vs_recordings.csv",
        summary_arr,
        delimiter=",",
        header="N,best_K,min_BIC",
        comments="",
        fmt="%.2f",
    )

    # 完整 BIC 表：每行一个 K，每列一个 N
    full_header = ["K"] + [str(N) for N in range(1, n_total + 1)]
    full_rows = []
    for ki, k in enumerate(k_list):
        row = [k] + [full_results[N][ki][3] for N in range(1, n_total + 1)]
        full_rows.append(row)
    np.savetxt(
        out_dir / "bic_full.csv",
        np.array(full_rows),
        delimiter=",",
        header=",".join(full_header),
        comments="",
        fmt="%.2f",
    )

    # 6. 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = [s[0] for s in summary]
    best_ks = [s[1] for s in summary]
    ax.plot(Ns, best_ks, "o-")
    ax.set_xlabel("Number of recordings")
    ax.set_ylabel("BIC best K")
    ax.set_title("BIC best K vs. recording count")
    ax.set_xticks(Ns)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "bic_vs_recordings.png", dpi=150)
    plt.close(fig)

    print(f"\n结果已保存到 {out_dir}")
    print("N -> best_K:")
    for N, best_k, bic in summary:
        print(f"  {int(N):2d}: K={int(best_k)}  (BIC={bic:.1f})")


if __name__ == "__main__":
    main()