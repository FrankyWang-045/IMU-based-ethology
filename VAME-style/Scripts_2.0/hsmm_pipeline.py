"""
HSMM 流程唯一入口。
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import hsmm_core as core
import hsmm_models as models
import hsmm_selection as sel


def output_dir(cfg: dict) -> Path:
    return Path(cfg["project"]["output_dir"])


def latest_experiment_dir(cfg: dict) -> Path:
    out = output_dir(cfg)
    exp_dirs = sorted([p for p in out.iterdir() if p.is_dir() and p.name.startswith("exp_")])
    if not exp_dirs:
        raise FileNotFoundError("未找到 exp_ 目录")
    return exp_dirs[-1]


def resolve_exp_dir(cfg: dict, exp_arg: str | None) -> Path:
    out = output_dir(cfg)
    if exp_arg:
        exp_dir = Path(exp_arg)
        if not exp_dir.is_absolute():
            exp_dir = out / exp_dir
    else:
        exp_dir = latest_experiment_dir(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def load_recordings(exp_dir: Path, rec_filter: list[str] | None) -> tuple[list[np.ndarray], list[dict]]:
    reader = core.LatentReader(exp_dir)
    files = reader.find_files(rec_filter)
    if not files:
        raise FileNotFoundError("未找到 *_latent.npz 文件")
    Xs = []
    metas = []
    for p in files:
        d = reader.load(p)
        Xs.append(d["X"])
        metas.append(d)
    return Xs, metas


def cmd_train(args: argparse.Namespace):
    cfg = core.load_config(args.config)
    hc = core.get_hmm_config(cfg)
    exp_dir = resolve_exp_dir(cfg, args.exp)

    Xs, _ = load_recordings(exp_dir, args.recs)
    X_all = np.concatenate(Xs, axis=0)

    fs = cfg["preprocess"]["target_fs"]
    d_max = int(hc.get("hsmm_d_max_sec", 5.0) * fs)
    min_d = hc.get("min_duration_frames", int(0.1 * fs))
    K = hc["n_states"]

    # 标准化
    scaler = core.StandardScaler()
    X_all = scaler.fit_transform(X_all)
    scaler.save(exp_dir / "hsmm_scaler.json")

    # 固定长度采样
    L = core.get_selection_config(cfg).get("fixed_length_lstar", min(10000, len(X_all)))
    sampler = core.FixedLengthSampler(length=L, seed=hc.get("hsmm_seed", 0))
    X_train = sampler.sample(X_all)

    model = models.HSMM(K, X_train.shape[1], d_max, duration_kind="nb")
    result = model.fit(
        X_train,
        n_iter=hc.get("hsmm_n_iter", 20),
        min_d=min_d,
        n_inits=hc.get("hsmm_n_init", 5),
        seed=hc.get("hsmm_seed", 0),
    )

    with open(exp_dir / "hsmm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "result": result, "scaler": scaler}, f)

    # 解码
    decode_log_P = model.get_decode_log_P(
        beta=hc.get("hsmm_sticky_beta", 0.0), min_d=min_d
    )
    for meta in load_recordings(exp_dir, args.recs)[1]:
        X = scaler.transform(meta["X"])
        states = model.viterbi(X, log_P=decode_log_P)
        np.savez(
            exp_dir / f"{meta['stem']}_hsmm.npz",
            states=states,
            time=meta["time"],
            interp_mask=meta["interp_mask"],
        )

    core.logger.info("HSMM 训练完成")


def cmd_scan(args: argparse.Namespace):
    cfg = core.load_config(args.config)
    hc = core.get_hmm_config(cfg)
    sc = core.get_selection_config(cfg)
    exp_dir = resolve_exp_dir(cfg, args.exp)

    Xs, _ = load_recordings(exp_dir, args.recs)
    X_all = np.concatenate(Xs, axis=0)
    scaler = core.StandardScaler()
    X_all = scaler.fit_transform(X_all)

    fs = cfg["preprocess"]["target_fs"]
    d_max = int(hc.get("hsmm_d_max_sec", 5.0) * fs)
    min_d = hc.get("min_duration_frames", int(0.1 * fs))
    L = sc.get("fixed_length_lstar", 10000)
    K_list = sc.get("k_list", list(range(5, 51, 5)))

    sampler = core.FixedLengthSampler(length=L, seed=hc.get("hsmm_seed", 0))
    X_sub = sampler.sample(X_all)

    scanner = sel.FixedLengthScanner(
        X_sub,
        K_list=K_list,
        d_max=d_max,
        n_iter=hc.get("hsmm_n_iter", 20),
        n_inits=sc.get("n_random_inits", 5),
        min_d=min_d,
        seed=hc.get("hsmm_seed", 0),
    )
    result = scanner.run()

    import pandas as pd
    df = pd.DataFrame(result.to_df_rows())
    df.to_csv(exp_dir / "scan_result.csv", index=False)
    with open(exp_dir / "scan_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_df_rows(), f, indent=2)

    core.logger.info(f"扫描完成，BIC 最优 K={result.best_bic_k}")


def cmd_diagnose(args: argparse.Namespace):
    cfg = core.load_config(args.config)
    hc = core.get_hmm_config(cfg)
    sc = core.get_selection_config(cfg)
    exp_dir = resolve_exp_dir(cfg, args.exp)

    Xs, _ = load_recordings(exp_dir, args.recs)
    X_all = np.concatenate(Xs, axis=0)
    scaler = core.StandardScaler()
    X_all = scaler.fit_transform(X_all)

    fs = cfg["preprocess"]["target_fs"]
    d_max = int(hc.get("hsmm_d_max_sec", 5.0) * fs)
    min_d = hc.get("min_duration_frames", int(0.1 * fs))
    L = sc.get("fixed_length_lstar", 10000)
    K_list = sc.get("k_list", list(range(5, 51, 5)))

    sampler = core.FixedLengthSampler(length=L, seed=hc.get("hsmm_seed", 0))
    X_sub = sampler.sample(X_all)

    # 复用扫描结果，若存在则加载
    scan_path = exp_dir / "scan_result.csv"
    if scan_path.exists():
        import pandas as pd
        df = pd.read_csv(scan_path)
        scan = sel.ScanResult(
            K_list=df["K"].tolist(),
            logliks=df["loglik"].tolist(),
            bics=df["BIC"].tolist(),
            aics=df["AIC"].tolist(),
            icls=df["ICL"].tolist(),
            best_bic_k=int(df.loc[df["BIC"].idxmin(), "K"]),
            best_aic_k=int(df.loc[df["AIC"].idxmin(), "K"]),
            best_icl_k=int(df.loc[df["ICL"].idxmin(), "K"]),
        )
    else:
        scan = sel.FixedLengthScanner(
            X_sub, K_list, d_max, seed=hc.get("hsmm_seed", 0)
        ).run()

    diag_dir = exp_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    reports = []

    # 嵌套检验：相邻 K
    fits = {}
    for K in K_list:
        model = models.HSMM(K, X_sub.shape[1], d_max)
        fits[K] = model.fit(X_sub, n_iter=10, min_d=min_d, seed=hc.get("hsmm_seed", 0))

    for i in range(len(K_list) - 1):
        k1, k2 = K_list[i], K_list[i + 1]
        nt = sel.NestedTest(fits[k1], fits[k2])
        reports.append(nt.report())

    # 占有率与时长
    for K, fit in fits.items():
        reports.append(sel.OccupancyTest(fit, threshold=sc.get("occupancy_threshold", 0.01)).report())
        reports.append(sel.DurationTailTest(fit, d_max=d_max).report())

    # 自助法（对 BIC 最优 K）
    best_k = scan.best_bic_k
    boot = sel.ParametricBootstrap(
        K=best_k,
        d_max=d_max,
        n_dims=X_sub.shape[1],
        length_grid=sc.get("length_grid", [1000, 2000, 4000, 8000]),
        repeats=sc.get("bootstrap_runs", 20),
        seed=hc.get("hsmm_seed", 0),
    )
    reports.append(boot.run(fits[best_k]))

    # 稳定性选择
    stab = sel.StabilitySelection(
        X_sub,
        K_list=K_list,
        d_max=d_max,
        L=L,
        n_bootstrap=sc.get("bootstrap_runs", 100),
        seed=hc.get("hsmm_seed", 0),
    )
    reports.append(stab.run())

    # 保存
    for r in reports:
        with open(diag_dir / f"{r.name}.json", "w", encoding="utf-8") as f:
            json.dump(r.data, f, indent=2, default=str)

    core.logger.info("诊断完成")


def cmd_report(args: argparse.Namespace):
    cfg = core.load_config(args.config)
    exp_dir = resolve_exp_dir(cfg, args.exp)
    scan_path = exp_dir / "scan_result.csv"
    if not scan_path.exists():
        raise FileNotFoundError("请先运行 scan")
    import pandas as pd
    df = pd.read_csv(scan_path)

    report_path = exp_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# HSMM 模型选择报告\n\n")
        f.write(f"## 扫描结果\n\n")
        f.write(df.to_markdown(index=False))
        f.write(f"\n\n- BIC 最优 K: {df.loc[df['BIC'].idxmin(), 'K']}\n")
        f.write(f"- ICL 最优 K: {df.loc[df['ICL'].idxmin(), 'K']}\n")

    core.logger.info(f"报告已保存: {report_path}")


def cmd_hdp(args: argparse.Namespace):
    cfg = core.load_config(args.config)
    hc = core.get_hmm_config(cfg)
    sc = core.get_selection_config(cfg)
    exp_dir = resolve_exp_dir(cfg, args.exp)

    Xs, metas = load_recordings(exp_dir, args.recs)
    X_all = np.concatenate(Xs, axis=0)
    scaler = core.StandardScaler()
    X_all = scaler.fit_transform(X_all)

    fs = cfg["preprocess"]["target_fs"]
    d_max = int(hc.get("hsmm_d_max_sec", 5.0) * fs)
    min_d = hc.get("min_duration_frames", int(0.1 * fs))
    K_max = hc["n_states"]

    model = models.HDP_HSMM(
        K_max=K_max,
        n_dims=X_all.shape[1],
        d_max=d_max,
        alpha=sc.get("hdp_alpha", 1.0),
        gamma=sc.get("hdp_gamma", 1.0),
    )
    result = model.fit(
        X_all,
        n_iter=sc.get("hdp_mcmc_iter", 200),
        min_d=min_d,
        seed=hc.get("hsmm_seed", 0),
    )

    with open(exp_dir / "hdp_hsmm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "result": result, "scaler": scaler}, f)

    for meta in metas:
        X = scaler.transform(meta["X"])
        states = model._sample_states(X, min_d=min_d)
        np.savez(
            exp_dir / f"{meta['stem']}_hdp_hsmm.npz",
            states=states,
            time=meta["time"],
            interp_mask=meta["interp_mask"],
        )

    core.logger.info(f"HDP-HSMM 完成，占用状态数: {result.params['occupied_states']}")


def main():
    default_config = Path(__file__).resolve().parent.parent / "config.yaml"

    parser = argparse.ArgumentParser(description="HSMM / HDP-HSMM 流程入口")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="配置文件路径，默认使用项目根目录 config.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "scan", "diagnose", "report", "hdp"]:
        p = sub.add_parser(name, help=f"{name} 命令")
        p.add_argument("--exp", type=str, default=None)
        p.add_argument("--recs", type=str, default=None)

    args = parser.parse_args()

    if args.recs:
        args.recs = [r.strip() for r in args.recs.split(",") if r.strip()]
    else:
        args.recs = None

    dispatch = {
        "train": cmd_train,
        "scan": cmd_scan,
        "diagnose": cmd_diagnose,
        "report": cmd_report,
        "hdp": cmd_hdp,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()