"""CategoricalHMM 行为分割：全体文件合并训练一个模型，逐文件解码保存。

观测 = VQ 码序列（离散），发射 = 每状态的码概率表。
"""

import pickle

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from utils import load_config, output_dir, hmm_path

MODEL_NAME = "hmm_cat_model.pkl"

def fit_categorical(X, lengths, n_states, n_seeds, n_iter, sticky=1.0):
    transmat_prior = np.full((n_states, n_states), 1.0)
    np.fill_diagonal(transmat_prior, sticky)

    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        m = CategoricalHMM(n_components=n_states, n_iter=n_iter,
                           transmat_prior=transmat_prior,
                           random_state=seed, verbose=False)
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                m.fit(X, lengths=lengths)                # ← 传入 lengths
            if not (np.isfinite(m.startprob_).all()
                    and np.isfinite(m.transmat_).all()
                    and np.isfinite(m.emissionprob_).all()):
                print(f"  seed={seed}  退化，跳过")
                continue
            score = m.score(X, lengths=lengths)          # ← 同样传入
        except ValueError:
            print(f"  seed={seed}  失败，跳过")
            continue
        print(f"  seed={seed}  loglik={score:.1f}")
        if score > best_score:
            best_score, best_model = score, m
    if best_model is None:
        raise RuntimeError(f"K={n_states} 所有 seed 均训练失败")
    print(f"最佳对数似然: {best_score:.1f}")
    return best_model

def enforce_min_duration(states, min_len=10):
    """短于 min_len 帧的状态段并入前一个段（首段则并入后一个）。

    循环处理直到没有短段（合并可能产生新的短段，故需迭代）。
    """
    states = states.copy()
    while True:
        changes = np.where(np.diff(states) != 0)[0]
        bounds = np.concatenate([[0], changes + 1, [len(states)]])
        durations = np.diff(bounds)
        short = np.where(durations < min_len)[0]
        if len(short) == 0:
            break
        i = short[0]                        # 每次只处理一个，避免连锁错误
        start, end = bounds[i], bounds[i + 1]
        if i == 0:                          # 首段 → 并入后段
            states[start:end] = states[end]
        else:                               # 其余 → 并入前段
            states[start:end] = states[start - 1]
    return states

def main():
    cfg = load_config()
    hc = cfg["hmm"]
    n_states = hc["n_states"]

    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError("没有 latent 文件，请先运行 embed.py")

    # ---------- 训练集：每文件随机抽连续段，lengths 标记段边界 ----------
    n_segments = hc.get("train_segments", 5)
    seg_len = hc.get("segment_len", 10000)
    rng = np.random.default_rng(cfg["train"]["seed"])

    segments, lengths = [], []
    for p in latent_files:
        codes = np.load(p)["codes"].astype(int)
        for _ in range(n_segments):
            start = rng.integers(0, len(codes) - seg_len)
            segments.append(codes[start:start + seg_len])
            lengths.append(seg_len)

    X_train = np.concatenate(segments).reshape(-1, 1)
    print(f"训练集: {len(segments)} 段 × {seg_len} 帧 = {len(X_train)} 帧, K={n_states}")

    model = fit_categorical(X_train, lengths, n_states,
                            hc["n_seeds"], hc["n_iter"],
                            sticky=hc.get("sticky", 10.0))

    with open(out / MODEL_NAME, "wb") as f:
        pickle.dump(model, f)

    # ---------- 解码：逐文件全量 + 最小时长约束 ----------
    min_len = hc.get("min_duration", 10)
    for p in latent_files:
        data = np.load(p)
        X = data["codes"].astype(int).reshape(-1, 1)
        states = model.predict(X)
        states = enforce_min_duration(states, min_len=min_len)
        probs = model.predict_proba(X)

        rec = p.stem.replace("_latent", "")
        np.savez(hmm_path(cfg, rec),
                 states=states, probs=probs,
                 time=data["time"], interp_mask=data["interp_mask"])

        occ = np.bincount(states, minlength=n_states) / len(states)
        print(f"  {rec}: 最大状态占比 {occ.max():.1%}, "
              f"<1% 的状态数 {(occ < 0.01).sum()}")

    top_codes = model.emissionprob_.argsort(axis=1)[:, ::-1][:, :3]
    print("\n各状态的高频码 (top3):")
    for s in range(n_states):
        print(f"  状态 {s}: {top_codes[s].tolist()}")


if __name__ == "__main__":
    main()