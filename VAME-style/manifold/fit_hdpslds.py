import argparse
import sys
import time

print(sys.executable)

import numpy as np

from pybasicbayes.distributions import Gaussian, Regression
from pyhsmm.basic.distributions import (
    Delay,
    NegativeBinomialFixedRDuration,
)
from pyslds.models import WeakLimitHDPHSMMSLDS
from pyslds.states import HSMMSLDSStatesEigen

from utils import load_config, output_dir

class CompatibleHSMMSLDSStates(HSMMSLDSStatesEigen):
    """
    兼容 pyslds 0.0.3 与 pyhsmm 的 HSMM 初始化参数。

    旧版 pyslds 的多重继承链没有把 right_censoring、
    left_censoring 和 trunc 正确传给 HSMMStatesPython。
    """

    def __init__(
        self,
        model,
        right_censoring=True,
        left_censoring=False,
        trunc=None,
        **kwargs
    ):
        self.right_censoring = right_censoring
        self.left_censoring = left_censoring
        self.trunc = trunc

        # 不再把这些 HSMM 参数传给旧版 SLDS 初始化方法。
        super().__init__(model, **kwargs)

class CompatibleWeakLimitHDPHSMMSLDS(WeakLimitHDPHSMMSLDS):
    _states_class = CompatibleHSMMSLDSStates

def get_segment_lengths(states):
    """返回每个连续状态片段的长度。"""
    boundaries = np.flatnonzero(
        np.r_[True, states[1:] != states[:-1], True]
    )
    return np.diff(boundaries)

def fit_hdp_hsmm_slds(
    data,
    latent_dim,
    max_states=50,
    n_iter=150,
    alpha=3.0,
    gamma=6.0,
    min_duration=10,      # 单位：原始 100 Hz 帧
    mean_duration=60,     # 单位：原始 100 Hz 帧
    duration_r=3.0,
    max_duration=300,     # 单位：原始 100 Hz 帧
    downsample=4,         # 降采样因子，1 = 不降采样
    param_every=5,        # 每隔多少轮重采样一次参数（状态每轮都重采样）
    seed=0,
):
    """
    带显式时长分布的 Weak Limit HDP-HSMM-SLDS（加速版）。

    时长参数均以原始 100 Hz 帧为单位传入，内部自动按 downsample 换算。
    downsample=4 时数据降到 25 Hz，T 缩小 4 倍， Kalman 似然与消息传递
    的成本同步下降。
    """
    if downsample < 1:
        raise ValueError("downsample 必须大于等于 1")
    if param_every < 1:
        raise ValueError("param_every 必须大于等于 1")

    if downsample > 1:
        data = np.ascontiguousarray(data[::downsample])

    T = len(data)

    # 把 100 Hz 帧单位的时长约束换算为降采样后的帧。
    min_dur = max(2, int(round(min_duration / downsample)))
    mean_dur = max(min_dur + 1, int(round(mean_duration / downsample)))
    max_dur = max(min_dur, int(round(max_duration / downsample)))

    if T < min_dur:
        raise ValueError("降采样后数据长度小于最短时长约束")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    eye = np.eye(latent_dim)

    # 每个离散状态的初始连续状态分布。
    init_dynamics_distns = []

    for _ in range(max_states):
        distn = Gaussian(
            mu_0=np.zeros(latent_dim),
            sigma_0=3.0 * eye,
            kappa_0=0.01,
            nu_0=latent_dim + 3,
        )
        distn.mu = np.zeros(latent_dim)
        distn.sigma = 0.01 * eye
        init_dynamics_distns.append(distn)

    # 状态相关的线性动态分布。
    dynamics_distns = []

    # 帧间持续性先验随降采样等效缩放：100 Hz 下 0.95/帧
    # 对应 25 Hz 下 0.95**4/帧。
    base_decay = 0.95 ** downsample

    for _ in range(max_states):
        A_linear = base_decay * eye + 0.01 * rng.standard_normal(
            (latent_dim, latent_dim)
        )

# affine=True 时，A 的最后一列是偏置 b，初始设为零。
        A = np.hstack([
            A_linear,
            np.zeros((latent_dim, 1)),
        ])

        dynamics_distns.append(
            Regression(
                nu_0=latent_dim + 1,
                S_0=latent_dim * eye,
                M_0=np.zeros((latent_dim, latent_dim + 1)),
                K_0=latent_dim * np.eye(latent_dim + 1),
                affine=True,
                A=A,
                sigma=0.01 * eye,
            )
        )

    # 共享一个发射分布（C = I, sigma = 0.001 I）：
    # 观测就是 latent 本身，发射在所有状态间完全相同，
    # pyslds 检测到非 list 输入后走 _single_emission 路径，
    # 发射似然只计算一次并广播到所有状态，而不是 K 次。
    shared_emission = Regression(
        nu_0=latent_dim + 1,
        S_0=latent_dim * eye,
        M_0=np.zeros((latent_dim, latent_dim)),
        K_0=latent_dim * eye,
        A=eye,
        sigma=0.001 * eye,
    )

    # 总时长：duration = min_dur + extra_duration，
    # extra_duration 服从负二项分布，mean = r * p / (1 - p)。
    extra_mean = mean_dur - min_dur
    p = extra_mean / (duration_r + extra_mean)

    prior_strength = 30.0
    duration_alpha_0 = prior_strength * p
    duration_beta_0 = prior_strength * (1.0 - p)

    dur_distns = [
        Delay(
            NegativeBinomialFixedRDuration(
                r=duration_r,
                p=p,
                alpha_0=duration_alpha_0,
                beta_0=duration_beta_0,
            ),
            delay=min_dur - 1,
        )
        for _ in range(max_states)
    ]

    model = CompatibleWeakLimitHDPHSMMSLDS(
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=shared_emission,   # 单个实例，不要列表
        dur_distns=dur_distns,
        alpha=alpha,
        gamma=gamma,
        init_state_distn="uniform",
    )

    # right_censoring=False：最后一个片段也满足完整时长约束。
    model.add_data(
        data,
        trunc=min(max_dur, T),
        right_censoring=False,
    )

    model.resample_states()

    initial_states = model.states_list[0].stateseq
    initial_lengths = get_segment_lengths(initial_states)

    if initial_lengths.min() < min_dur:
        raise RuntimeError(
            "初始化后仍存在小于最短约束的片段："
            f"{initial_lengths.min()} 帧"
        )

    print(
        f"开始 HDP-HSMM-SLDS 推断，"
        f"max_states={max_states}, n_iter={n_iter}, "
        f"降采样={downsample} (T={T}), "
        f"min_duration={min_dur}, mean_duration={mean_dur}, "
        f"max_duration={max_dur}, 参数重采样间隔={param_every}"
    )

    t_start = time.perf_counter()

    for itr in range(n_iter):
        if itr % param_every == 0 or itr == n_iter - 1:
            model.resample_model()       # 状态 + 参数
        else:
            model.resample_states()      # 只重采样状态

        if itr % 10 == 0 or itr == n_iter - 1:
            states = model.states_list[0].stateseq
            lengths = get_segment_lengths(states)
            elapsed = time.perf_counter() - t_start

            print(
                f"  iter {itr:4d}/{n_iter}: "
                f"有效状态数 = {len(np.unique(states)):3d}, "
                f"片段数 = {len(lengths):4d}, "
                f"最短段长 = {lengths.min():3d}, "
                f"中位段长 = {np.median(lengths):6.1f}, "
                f"累计耗时 {elapsed:7.1f}s"
            )

    states = model.states_list[0].stateseq
    return states, model

def main():
    parser = argparse.ArgumentParser(
        description="带最短段长约束的 HDP-HSMM-SLDS（加速版）"
    )

    parser.add_argument("--recording", required=True, help="recording 名")
    parser.add_argument(
        "--max_states", type=int, default=50, help="最大状态数"
    )
    parser.add_argument(
        "--n_iter", type=int, default=100, help="Gibbs 采样迭代次数"
    )
    parser.add_argument(
        "--alpha", type=float, default=3.0, help="HDP 状态转移浓度参数"
    )
    parser.add_argument(
        "--gamma", type=float, default=6.0, help="HDP 全局状态分布浓度参数"
    )
    parser.add_argument(
        "--min_duration", type=int, default=25,
        help="最短片段长度，单位为原始 100 Hz 帧",
    )
    parser.add_argument(
        "--mean_duration", type=float, default=60,
        help="初始平均片段长度，单位为原始 100 Hz 帧",
    )
    parser.add_argument(
        "--duration_r", type=float, default=3.0,
        help="时长分布形状参数，越小则分布越宽",
    )
    parser.add_argument(
        "--max_duration", type=int, default=300,
        help="推断时的最大时长截断，单位为原始 100 Hz 帧",
    )
    parser.add_argument(
        "--downsample", type=int, default=2,
        help="降采样因子，2 = 100 Hz 降到 50 Hz，1 = 不降采样",
    )
    parser.add_argument(
        "--param_every", type=int, default=5,
        help="每隔多少轮重采样一次模型参数（状态每轮都重采样）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")

    args = parser.parse_args()

    if args.max_states < 1:
        parser.error("--max_states 必须大于等于 1")
    if args.n_iter < 1:
        parser.error("--n_iter 必须大于等于 1")
    if args.alpha <= 0:
        parser.error("--alpha 必须大于 0")
    if args.gamma <= 0:
        parser.error("--gamma 必须大于 0")
    if args.min_duration < 1:
        parser.error("--min_duration 必须大于等于 1")
    if args.mean_duration <= args.min_duration:
        parser.error("--mean_duration 必须大于 --min_duration")
    if args.max_duration < args.min_duration:
        parser.error("--max_duration 不能小于 --min_duration")
    if args.duration_r <= 0:
        parser.error("--duration_r 必须大于 0")
    if args.downsample < 1:
        parser.error("--downsample 必须大于等于 1")
    if args.param_every < 1:
        parser.error("--param_every 必须大于等于 1")

    cfg = load_config()
    out = output_dir(cfg)
    latent_dir = out / "latent"
    npz_path = latent_dir / f"{args.recording}_latent.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 latent: {npz_path}")

    data = np.load(npz_path)
    latent = data["latent"].astype(np.float64)
    time_full = data["time"]

    if latent.ndim != 2:
        raise ValueError(
            f"latent 应为二维数组，实际形状为 {latent.shape}"
        )
    if len(latent) < args.min_duration:
        raise ValueError("数据长度小于 min_duration")
    if not np.isfinite(latent).all():
        raise ValueError("latent 中包含 NaN 或 Inf")

    states, model = fit_hdp_hsmm_slds(
        latent,
        latent_dim=latent.shape[1],
        max_states=args.max_states,
        n_iter=args.n_iter,
        alpha=args.alpha,
        gamma=args.gamma,
        min_duration=args.min_duration,
        mean_duration=args.mean_duration,
        duration_r=args.duration_r,
        max_duration=args.max_duration,
        downsample=args.downsample,
        param_every=args.param_every,
        seed=args.seed,
    )

    lengths = get_segment_lengths(states)

    if lengths.min() * args.downsample < args.min_duration:
        raise RuntimeError(
            "发现小于最短约束的片段："
            f"{lengths.min()} 帧（降采样后）"
        )

    # 状态序列与 latent 等长（均为降采样后），time 同步降采样。
    output_path = latent_dir / f"{args.recording}_hsmm_states.npz"

    np.savez_compressed(
        output_path,
        states=states.astype(np.int32),
        time=time_full[::args.downsample],
        latent=latent[::args.downsample].astype(np.float32),
        segment_lengths=lengths.astype(np.int32),
        downsample=args.downsample,
        min_duration=args.min_duration,
        mean_duration=args.mean_duration,
        duration_r=args.duration_r,
        max_duration=args.max_duration,
        alpha=args.alpha,
        gamma=args.gamma,
        max_states=args.max_states,
        n_iter=args.n_iter,
        param_every=args.param_every,
        seed=args.seed,
    )

    print(f"状态标签已保存: {output_path}")
    print(f"  有效状态数: {len(np.unique(states))}")
    print(f"  片段数: {len(lengths)}")
    print(f"  最短段长: {lengths.min()} 帧（降采样后）")
    print(f"  中位段长: {np.median(lengths):.1f} 帧（降采样后）")
    print(f"  平均段长: {lengths.mean():.1f} 帧（降采样后）")

if __name__ == "__main__":
    main()