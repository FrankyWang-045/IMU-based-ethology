"""第二层 Categorical HMM：对第一层 HSMM 的 state 序列做高阶行为分割。

输入：exp_xxx/*_hsmm.npz 中的离散 state 序列
输出（直接在 exp_xxx/ 根目录）：
    - *_hmm2_{suffix}.npz
    - hmm2_{suffix}_params.npz
    - hmm2_{suffix}_stats.json
    - hmm2_{suffix}_report.txt
    - hmm2_{suffix}_vis.png
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

from utils import load_config, output_dir, latest_experiment_dir


def resolve_exp_dir(args_exp):
    """根据 --exp 参数或最新实验目录确定输出目录。"""
    cfg = load_config()
    out = output_dir(cfg)

    if args_exp:
        p = Path(args_exp)
        if p.is_absolute():
            return p
        return out / args_exp

    return latest_experiment_dir(cfg)


def load_first_layer_states(exp_dir):
    """加载第一层 HSMM 的所有 state 序列。"""
    exp_dir = Path(exp_dir)
    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))

    recs, state_seqs, time_seqs = [], [], []
    for p in hsmm_files:
        data = np.load(p)
        states = data["states"].astype(np.int64)
        time = data["time"]

        if len(states) == 0:
            print(f"  跳过空序列: {p.stem}")
            continue

        rec_name = p.stem.replace("_hsmm", "")
        recs.append(rec_name)
        state_seqs.append(states)
        time_seqs.append(time)

    return recs, state_seqs, time_seqs


def remap_to_contiguous(state_seqs):
    """把非连续的 state id 映射到 0..K-1。"""
    unique = sorted({int(s) for seq in state_seqs for s in seq})
    forward = {old: new for new, old in enumerate(unique)}
    backward = {new: old for old, new in forward.items()}

    remapped = [np.array([forward[int(s)] for s in seq], dtype=np.int64)
                for seq in state_seqs]
    return remapped, forward, backward


def concatenate_sequences(state_seqs):
    """把多个序列拼接成 hmmlearn 需要的格式。"""
    lengths = [len(seq) for seq in state_seqs]
    X = np.concatenate(state_seqs).reshape(-1, 1)
    return X, lengths


def apply_sticky_transmat(model, sticky_self_loop):
    """把转移矩阵自环概率压到 sticky_self_loop，用于解码。"""
    K = model.transmat_.shape[0]
    off_diag = (1 - sticky_self_loop) / (K - 1) if K > 1 else 0.0
    sticky = np.eye(K) * sticky_self_loop + (1 - np.eye(K)) * off_diag
    new_transmat = model.transmat_ * (1 - sticky_self_loop) + sticky * sticky_self_loop
    new_transmat = new_transmat / new_transmat.sum(axis=1, keepdims=True)
    model.transmat_ = new_transmat
    return model


def fit_categorical_hmm(X, lengths, n_states, n_categories, n_iter, n_init,
                        sticky_self_loop, random_state):
    """训练 CategoricalHMM，多初始化选最优。"""
    best_model = None
    best_score = -np.inf
    rs = np.random.RandomState(random_state)

    for i in range(n_init):
        model = hmm.CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            random_state=rs.randint(0, 2**31 - 1),
            params="ste",
            init_params="ste",
            tol=1e-3,
            verbose=False,
        )

        try:
            model.fit(X, lengths=lengths)
        except Exception as e:
            print(f"  init {i+1}/{n_init} 失败: {e}")
            continue

        score = model.score(X, lengths=lengths)
        print(f"  init {i+1}/{n_init}: loglik={score:.2f}")
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("所有初始化都失败")

    print(f"最优 loglik: {best_score:.2f}")

    if sticky_self_loop > 0 and n_states > 1:
        best_model = apply_sticky_transmat(best_model, sticky_self_loop)
        print(f"已应用 sticky 解码先验: self-loop={sticky_self_loop:.3f}")

    return best_model, best_score

def decode_sequences(model, state_seqs):
    action_seqs, logliks = [], []
    for seq in state_seqs:
        X = seq.reshape(-1, 1)
        loglik, actions = model.decode(X, algorithm="viterbi")
        action_seqs.append(actions.astype(np.int64))
        logliks.append(loglik)
    return action_seqs, logliks


def compute_action_stats(action_seqs, time_seqs, recs, fs):
    """计算每个 recording 的 action 统计量。"""
    stats = []
    for rec, actions, time in zip(recs, action_seqs, time_seqs):
        T = len(actions)
        n_actions = int(actions.max()) + 1
        occupancy = np.bincount(actions, minlength=n_actions) / T

        # 片段时长
        durations = {a: [] for a in range(n_actions)}
        t = 0
        while t < T:
            s = t
            a = int(actions[t])
            while t < T and actions[t] == a:
                t += 1
            durations[a].append((t - s) / fs)

        mean_dur = {
            a: (np.mean(d) if d else 0.0)
            for a, d in durations.items()
        }
        median_dur = {
            a: (float(np.median(d)) if d else 0.0)
            for a, d in durations.items()
        }
        n_segments = {
            a: len(d) for a, d in durations.items()
        }

        stats.append({
            "recording": rec,
            "n_frames": T,
            "duration_sec": (time[-1] - time[0]) if len(time) > 1 else 0.0,
            "occupancy": occupancy.tolist(),
            "mean_duration_sec": mean_dur,
            "median_duration_sec": median_dur,
            "n_segments": n_segments,
        })
    return stats


def build_report(exp_dir, suffix, n_rec, n_fl_states, n_actions,
                 model, stats, best_score, sticky_self_loop):
    """生成文字报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("第二层 Categorical HMM 训练报告")
    lines.append("=" * 60)
    lines.append(f"实验目录: {exp_dir}")
    lines.append(f"输出后缀: {suffix}")
    lines.append(f"recording 数: {n_rec}")
    lines.append(f"第一层 state 数: {n_fl_states}")
    lines.append(f"第二层 action 数: {n_actions}")
    lines.append(f"训练 loglik: {best_score:.2f}")
    lines.append(f"sticky self-loop: {sticky_self_loop:.3f}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("Action 转移矩阵（自环已加 sticky）")
    lines.append("-" * 60)
    for i, row in enumerate(model.transmat_):
        lines.append(f"  action {i:2d}: " + " ".join([f"{x:.3f}" for x in row]))
    lines.append("")

    lines.append("-" * 60)
    lines.append("每 recording 的 action 占比与时长")
    lines.append("-" * 60)
    for s in stats:
        lines.append(f"recording: {s['recording']}")
        lines.append(f"  总时长: {s['duration_sec']:.1f}s, 帧数: {s['n_frames']}")
        occ = np.array(s["occupancy"])
        for a in range(len(occ)):
            lines.append(
                f"  action {a:2d}: 占比 {occ[a]:.3f}, "
                f"片段数 {s['n_segments'][a]}, "
                f"中位时长 {s['median_duration_sec'][a]:.2f}s"
            )
        lines.append("")

    return "\n".join(lines)


def save_report(exp_dir, suffix, report_text):
    report_path = Path(exp_dir) / f"hmm2_{suffix}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告保存: {report_path}")


def visualize_segments(exp_dir, suffix, rec, t0, window, fs):
    """可视化第一层 state 与第二层 action 对照图。"""
    exp_dir = Path(exp_dir)
    hmm_h = np.load(exp_dir / f"{rec}_hsmm.npz")
    states_h, time_h = hmm_h["states"], hmm_h["time"]

    hmm2 = np.load(exp_dir / f"{rec}_hmm2_{suffix}.npz")
    actions, time_a = hmm2["actions"], hmm2["time"]

    n_h = int(states_h.max()) + 1
    n_a = int(actions.max()) + 1

    t1 = t0 + window
    m_h = (time_h >= t0) & (time_h < t1)
    m_a = (time_a >= t0) & (time_a < t1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # 第一层 HSMM state
    axes[0].scatter(
        time_h[m_h], states_h[m_h],
        c=states_h[m_h], cmap="tab20", s=4,
        vmin=-0.5, vmax=n_h - 0.5,
    )
    axes[0].set_ylabel("first-layer state")
    axes[0].set_yticks(range(n_h))
    axes[0].set_ylim(-0.5, n_h - 0.5)
    axes[0].set_title(f"{rec} | first-layer HSMM vs second-layer action ({suffix})")

    # 第二层 action
    axes[1].scatter(
        time_a[m_a], actions[m_a],
        c=actions[m_a], cmap="tab20", s=6,
        vmin=-0.5, vmax=n_a - 0.5,
    )
    axes[1].set_ylabel("second-layer action")
    axes[1].set_yticks(range(n_a))
    axes[1].set_ylim(-0.5, n_a - 0.5)
    axes[1].set_xlabel("time (s)")

    fig.tight_layout()
    vis_path = exp_dir / f"hmm2_{suffix}_vis.png"
    fig.savefig(vis_path, dpi=150)
    plt.close(fig)
    print(f"可视化保存: {vis_path}")

def save_results(exp_dir, suffix, recs, action_seqs, time_seqs, model, stats):
    """保存 action-level 结果和模型参数。"""
    exp_dir = Path(exp_dir)

    for rec, actions, time in zip(recs, action_seqs, time_seqs):
        np.savez_compressed(
            exp_dir / f"{rec}_hmm2_{suffix}.npz",
            actions=actions,
            time=time,
        )

    np.savez_compressed(
        exp_dir / f"hmm2_{suffix}_params.npz",
        transmat=model.transmat_,
        startprob=model.startprob_,
        emissionprob=model.emissionprob_,
        n_states=model.n_components,
        n_categories=model.emissionprob_.shape[-1],
    )

    with open(exp_dir / f"hmm2_{suffix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {exp_dir}")

    # 模型参数
    np.savez_compressed(
        exp_dir / f"hmm2_{suffix}_params.npz",
        transmat=model.transmat_,
        startprob=model.startprob_,
        emissionprob=model.emissionprob_,
        n_states=model.n_components,
        n_categories=model.n_categories,
    )

    # 统计量
    with open(exp_dir / f"hmm2_{suffix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="第二层 Categorical HMM")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验目录，默认使用最新 exp")
    parser.add_argument("--k", type=int, default=12,
                        help="第二层 action state 数")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="EM 迭代次数")
    parser.add_argument("--n_init", type=int, default=10,
                        help="随机初始化次数")
    parser.add_argument("--sticky_self_loop", type=float, default=0.95,
                        help="解码时转移矩阵自环概率")
    parser.add_argument("--random_state", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--suffix", type=str, default="k12",
                        help="输出文件后缀，避免覆盖")
    parser.add_argument("--vis_rec", type=str, default=None,
                        help="可视化指定 recording，默认取第一个")
    parser.add_argument("--vis_t0", type=float, default=600.0,
                        help="可视化起始时间（秒）")
    parser.add_argument("--vis_window", type=float, default=15.0,
                        help="可视化窗口长度（秒）")
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(args.exp)
    cfg = load_config()
    fs = cfg["preprocess"]["target_fs"]

    print(f"实验目录: {exp_dir}")
    print(f"采样率: {fs} Hz")

    # 加载第一层
    recs, state_seqs, time_seqs = load_first_layer_states(exp_dir)
    if len(recs) == 0:
        raise FileNotFoundError(f"{exp_dir} 中没有 *_hsmm.npz 文件")
    print(f"加载 {len(recs)} 个 recording 的第一层 state 序列")

    # 映射连续
    state_seqs_mapped, forward_map, backward_map = remap_to_contiguous(state_seqs)
    n_fl_states = len(forward_map)
    print(f"第一层 state 类别数: {n_fl_states}")

    # 拼接
    X, lengths = concatenate_sequences(state_seqs_mapped)

    # 训练
    print(f"\n训练第二层 Categorical HMM (K={args.k})...")
    model, score = fit_categorical_hmm(
        X, lengths,
        n_states=args.k,
        n_categories=n_fl_states,
        n_iter=args.n_iter,
        n_init=args.n_init,
        sticky_self_loop=args.sticky_self_loop,
        random_state=args.random_state,
    )

    # 解码
    print("\n解码所有 recording...")
    action_seqs, logliks = decode_sequences(model, state_seqs_mapped)

    # 统计
    stats = compute_action_stats(action_seqs, time_seqs, recs, fs)
    for rec, ll in zip(recs, logliks):
        print(f"  {rec}: loglik={ll:.2f}")

    # 保存结果
    save_results(exp_dir, args.suffix, recs, action_seqs, time_seqs, model, stats)

    # 生成并保存报告
    report = build_report(
        exp_dir, args.suffix, len(recs), n_fl_states, args.k,
        model, stats, score, args.sticky_self_loop,
    )
    print("\n" + report)
    save_report(exp_dir, args.suffix, report)

    # 可视化
    vis_rec = args.vis_rec or recs[0]
    if vis_rec in recs:
        print(f"\n生成可视化: {vis_rec}")
        visualize_segments(exp_dir, args.suffix, vis_rec, args.vis_t0, args.vis_window, fs)
    else:
        print(f"警告: --vis_rec={vis_rec} 不存在，跳过可视化")

    print("\n完成")


if __name__ == "__main__":
    main()