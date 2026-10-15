"""HMM2 结果分析与可视化：读取已有的 *_hmm2_*.npz 生成报告和图。"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, output_dir, latest_experiment_dir


def resolve_exp_dir(args_exp):
    cfg = load_config()
    out = output_dir(cfg)
    if args_exp:
        p = Path(args_exp)
        return p if p.is_absolute() else out / args_exp
    return latest_experiment_dir(cfg)


def load_hmm2_results(exp_dir, suffix):
    """加载第二层 HMM 的 action 序列。"""
    files = sorted(exp_dir.glob(f"*_hmm2_{suffix}.npz"))
    recs, action_seqs, time_seqs = [], [], []
    for p in files:
        data = np.load(p)
        rec = p.stem.replace(f"_hmm2_{suffix}", "")
        recs.append(rec)
        action_seqs.append(data["actions"].astype(np.int64))
        time_seqs.append(data["time"])
    return recs, action_seqs, time_seqs


def load_first_layer(exp_dir):
    """加载第一层 HSMM 的 state 序列。"""
    files = sorted(exp_dir.glob("*_hsmm.npz"))
    recs, state_seqs, time_seqs = [], [], []
    for p in files:
        data = np.load(p)
        rec = p.stem.replace("_hsmm", "")
        recs.append(rec)
        state_seqs.append(data["states"].astype(np.int64))
        time_seqs.append(data["time"])
    return recs, state_seqs, time_seqs


def load_latent(exp_dir, rec):
    p = exp_dir / f"{rec}_latent.npz"
    if not p.exists():
        return None, None
    data = np.load(p)
    return data["downstream"], data["time"]


def duration_stats(states, fs):
    changes = np.where(np.diff(states) != 0)[0]
    durations = np.diff(np.concatenate([[0], changes, [len(states)]])) / fs
    if len(durations) == 0:
        return 0.0, 0, 0
    return float(np.median(durations)), int((durations < 0.1).sum()), int(len(durations))


def compute_stats(recs, action_seqs, time_seqs, fs):
    """计算每个 recording 的 action 统计量。"""
    stats = []
    for rec, actions, time in zip(recs, action_seqs, time_seqs):
        T = len(actions)
        n_actions = int(actions.max()) + 1
        occupancy = np.bincount(actions, minlength=n_actions) / T

        durations = {a: [] for a in range(n_actions)}
        t = 0
        while t < T:
            s = t
            a = int(actions[t])
            while t < T and actions[t] == a:
                t += 1
            durations[a].append((t - s) / fs)

        med_dur = {a: float(np.median(d)) if d else 0.0 for a, d in durations.items()}
        n_seg = {a: len(d) for a, d in durations.items()}

        med, short, total = duration_stats(actions, fs)

        stats.append({
            "recording": rec,
            "n_frames": T,
            "duration_sec": float(time[-1] - time[0]) if len(time) > 1 else 0.0,
            "occupancy": occupancy.tolist(),
            "median_duration_sec": med_dur,
            "n_segments": n_seg,
            "overall_median_duration_sec": med,
            "short_fragments": short,
            "total_segments": total,
        })
    return stats


def build_report(exp_dir, suffix, recs, action_seqs, fs, stats, params=None):
    lines = []
    lines.append("=" * 70)
    lines.append(f"第二层 HMM 分析报告  |  suffix={suffix}")
    lines.append("=" * 70)
    lines.append(f"实验目录: {exp_dir}")
    lines.append(f"recording 数: {len(recs)}")
    if params is not None:
        lines.append(f"action 数 (K): {int(params.get('n_states', 'N/A'))}")
        lines.append(f"第一层 state 类别数: {int(params.get('n_categories', 'N/A'))}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("每 recording 统计")
    lines.append("-" * 70)
    for s in stats:
        lines.append(f"recording: {s['recording']}")
        lines.append(f"  总时长: {s['duration_sec']:.1f}s | 帧数: {s['n_frames']}")
        lines.append(f"  中位片段时长: {s['overall_median_duration_sec']:.2f}s")
        lines.append(f"  <0.1s 碎片: {s['short_fragments']}/{s['total_segments']}")
        occ = np.array(s["occupancy"])
        for a in range(len(occ)):
            lines.append(
                f"  action {a:2d}: 占比 {occ[a]:.3f} | "
                f"片段数 {s['n_segments'][a]} | "
                f"中位时长 {s['median_duration_sec'][a]:.2f}s"
            )
        lines.append("")

    return "\n".join(lines)


def visualize_recording(exp_dir, suffix, rec, t0, window, fs, out_dir):
    """绘制单 recording 的第一层 state 与第二层 action 对照图。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hmm_h = np.load(exp_dir / f"{rec}_hsmm.npz")
    states_h, time_h = hmm_h["states"], hmm_h["time"]

    hmm2 = np.load(exp_dir / f"{rec}_hmm2_{suffix}.npz")
    actions, time_a = hmm2["actions"], hmm2["time"]

    z, time_z = load_latent(exp_dir, rec)

    n_h = int(states_h.max()) + 1
    n_a = int(actions.max()) + 1

    t1 = t0 + window
    m_h = (time_h >= t0) & (time_h < t1)
    m_a = (time_a >= t0) & (time_a < t1)

    n_panels = 3 if z is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)

    ax_idx = 0

    # latent
    if z is not None:
        m_z = (time_z >= t0) & (time_z < t1)
        for d in range(z.shape[1]):
            axes[ax_idx].plot(time_z[m_z], z[m_z, d], alpha=0.7, label=f"z{d}")
        axes[ax_idx].set_ylabel("latent")
        axes[ax_idx].legend(loc="upper right", ncol=3, fontsize=8)
        axes[ax_idx].set_title(f"{rec} | latent + HSMM + HMM2 ({suffix})")
        ax_idx += 1
    else:
        axes[ax_idx].set_title(f"{rec} | HSMM + HMM2 ({suffix})")

    # first-layer state
    axes[ax_idx].scatter(
        time_h[m_h], states_h[m_h],
        c=states_h[m_h], cmap="tab20", s=3,
        vmin=-0.5, vmax=n_h - 0.5,
    )
    axes[ax_idx].set_ylabel("first-layer state")
    axes[ax_idx].set_yticks(range(n_h))
    axes[ax_idx].set_ylim(-0.5, n_h - 0.5)
    ax_idx += 1

    # second-layer action
    axes[ax_idx].scatter(
        time_a[m_a], actions[m_a],
        c=actions[m_a], cmap="tab20", s=6,
        vmin=-0.5, vmax=n_a - 0.5,
    )
    axes[ax_idx].set_ylabel("second-layer action")
    axes[ax_idx].set_yticks(range(n_a))
    axes[ax_idx].set_ylim(-0.5, n_a - 0.5)
    axes[ax_idx].set_xlabel("time (s)")

    fig.tight_layout()
    out_path = out_dir / f"{rec}_hmm2_{suffix}_vis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  可视化保存: {out_path}")


def plot_summary_heatmap(exp_dir, suffix, recs, action_seqs, out_dir):
    """绘制所有 recording 的 action occupancy 热图。"""
    out_dir = Path(out_dir)
    n_recs = len(recs)
    if n_recs == 0:
        return

    n_actions = max(int(seq.max()) + 1 for seq in action_seqs)
    occupancy = np.zeros((n_recs, n_actions))
    for i, seq in enumerate(action_seqs):
        occupancy[i] = np.bincount(seq, minlength=n_actions) / len(seq)

    fig, ax = plt.subplots(figsize=(10, max(4, n_recs * 0.5)))
    im = ax.imshow(occupancy, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels([f"A{a}" for a in range(n_actions)])
    ax.set_yticks(range(n_recs))
    ax.set_yticklabels(recs, fontsize=8)
    ax.set_title(f"Action occupancy per recording ({suffix})")
    plt.colorbar(im, ax=ax, label="occupancy")
    fig.tight_layout()
    out_path = out_dir / f"hmm2_{suffix}_occupancy_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  热图保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="HMM2 结果分析与可视化")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验目录，默认最新 exp")
    parser.add_argument("--suffix", type=str, default="k12",
                        help="hmm2 结果后缀")
    parser.add_argument("--rec", type=str, default=None,
                        help="指定可视化的 recording，默认第一个")
    parser.add_argument("--t0", type=float, default=600.0,
                        help="可视化起始时间（秒）")
    parser.add_argument("--window", type=float, default=120.0,
                        help="可视化窗口长度（秒）")
    parser.add_argument("--all_recs", action="store_true",
                        help="为所有 recording 生成可视化")
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(args.exp)
    cfg = load_config()
    fs = cfg["preprocess"]["target_fs"]

    print(f"实验目录: {exp_dir}")
    print(f"读取 HMM2 后缀: {args.suffix}")

    # 创建报告输出文件夹
    report_dir = exp_dir / f"hmm2_report_{args.suffix}"
    report_dir.mkdir(exist_ok=True)
    print(f"报告输出: {report_dir}")

    # 加载数据
    recs2, action_seqs, time_seqs = load_hmm2_results(exp_dir, args.suffix)
    if len(recs2) == 0:
        raise FileNotFoundError(f"{exp_dir} 中没有 *_hmm2_{args.suffix}.npz")

    recs1, state_seqs, _ = load_first_layer(exp_dir)

    # 加载模型参数
    params_path = exp_dir / f"hmm2_{args.suffix}_params.npz"
    params = np.load(params_path) if params_path.exists() else None

    # 计算统计量
    stats = compute_stats(recs2, action_seqs, time_seqs, fs)

    # 生成报告
    report = build_report(exp_dir, args.suffix, recs2, action_seqs, fs, stats, params)
    print("\n" + report)

    report_path = report_dir / f"hmm2_{args.suffix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告保存: {report_path}")

    # 保存 stats
    with open(report_dir / f"hmm2_{args.suffix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 可视化
    print("\n生成可视化...")
    if args.all_recs:
        for rec in recs2:
            visualize_recording(exp_dir, args.suffix, rec, args.t0, args.window, fs, report_dir)
    else:
        rec = args.rec or recs2[0]
        visualize_recording(exp_dir, args.suffix, rec, args.t0, args.window, fs, report_dir)

    # summary 热图
    plot_summary_heatmap(exp_dir, args.suffix, recs2, action_seqs, report_dir)

    print("\n完成")


if __name__ == "__main__":
    main()