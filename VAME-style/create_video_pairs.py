"""
随机抽取视频对用于人工类别判断。

用法：
    python create_video_pairs.py --base-dir "F:\\...\\A1_C1_C1-1c"
"""

import argparse
import csv
import random
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

# 默认参数
VIDEO_EXT = ".mp4"           # 视频格式
TOTAL_PAIRS = 200            # 总对数
SAME_STATE_RATIO = 0.5       # 同 state 对比例
MIN_PAIRS_PER_STATE = 3      # 每个 state 在全部配对中至少出现的次数
SEED = 42                    # 随机种子


def pair_key(state_a, video_a, state_b, video_b):
    """返回与顺序无关的配对唯一标识。"""
    key_a = (state_a, str(video_a))
    key_b = (state_b, str(video_b))
    return tuple(sorted([key_a, key_b]))


def collect_states(base_dir, ext):
    """收集所有非空 state 文件夹及其视频文件。"""
    base = Path(base_dir)
    states = {}
    for item in sorted(base.iterdir()):
        if item.is_dir() and item.name != "record":
            videos = sorted(
                f for f in item.iterdir()
                if f.is_file() and f.suffix.lower() == ext.lower()
            )
            if videos:
                states[item.name] = videos
    return states


def create_video_pairs(base_dir, ext=VIDEO_EXT, total_pairs=TOTAL_PAIRS,
                       same_state_ratio=SAME_STATE_RATIO,
                       min_per_state=MIN_PAIRS_PER_STATE, seed=SEED):
    random.seed(seed)
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"目录不存在：{base}")

    record_dir = base / "record"
    record_dir.mkdir(exist_ok=True)

    states = collect_states(base, ext)
    if len(states) < 2:
        raise ValueError(
            f"至少需要 2 个有视频的 state 文件夹，当前只有 {len(states)} 个"
        )

    state_list = sorted(states.keys())
    same_target = int(total_pairs * same_state_ratio)
    cross_target = total_pairs - same_target

    # 可行性检查：每个 state 至少出现 min_per_state 次
    min_total_appearances = len(state_list) * min_per_state
    max_total_appearances = total_pairs * 2
    if min_total_appearances > max_total_appearances:
        raise ValueError(
            f"state 数量（{len(state_list)}）过多，无法满足每个 state 至少出现 "
            f"{min_per_state} 次（需要 {min_total_appearances} 次，最多 {max_total_appearances} 次）。"
        )

    # 生成候选配对
    all_same = []
    for state, videos in states.items():
        if len(videos) >= 2:
            for v1, v2 in combinations(videos, 2):
                all_same.append((state, v1, state, v2))

    all_cross = []
    for i, s1 in enumerate(state_list):
        for s2 in state_list[i + 1:]:
            for v1 in states[s1]:
                for v2 in states[s2]:
                    all_cross.append((s1, v1, s2, v2))

    if len(all_same) < same_target:
        raise ValueError(
            f"同 state 候选对不足：需要 {same_target}，只有 {len(all_same)}"
        )
    if len(all_cross) < cross_target:
        raise ValueError(
            f"跨 state 候选对不足：需要 {cross_target}，只有 {len(all_cross)}"
        )

    used = set()
    selected = []
    appearances = Counter()

    def add_pair(p):
        key = pair_key(*p)
        if key in used:
            return False
        used.add(key)
        selected.append(p)
        appearances[p[0]] += 1
        appearances[p[2]] += 1
        return True

    # 阶段 1：每个 state 尽量先分配同 state 对，但不超过同 state 总目标
    # 这样即使 state 数量很多，也不会把同 state 配额占满。
    same_budget = same_target
    base_same_per_state = min(min_per_state, same_budget // len(state_list))
    extra_same_slots = same_budget - base_same_per_state * len(state_list)

    for i, state in enumerate(state_list):
        videos = states[state]
        if len(videos) < 2:
            continue
        target = base_same_per_state + (1 if i < extra_same_slots else 0)
        state_same = [(state, v1, state, v2) for v1, v2 in combinations(videos, 2)]
        random.shuffle(state_same)
        allocated = 0
        for p in state_same:
            if allocated >= target:
                break
            if add_pair(p):
                allocated += 1

    # 阶段 2：补齐同 state 对到目标数量
    same_count = sum(1 for p in selected if p[0] == p[2])
    remaining_same = [p for p in all_same if pair_key(*p) not in used]
    need_same = same_target - same_count
    if need_same > 0:
        if len(remaining_same) < need_same:
            raise ValueError(f"同 state 配对不足，无法填充到 {same_target}")
        for p in random.sample(remaining_same, need_same):
            add_pair(p)

    # 阶段 3：用跨 state 对弥补仍有出现次数不足的 state
    for state in state_list:
        deficit = max(0, min_per_state - appearances[state])
        if deficit <= 0:
            continue
        candidates = [
            p for p in all_cross
            if state in (p[0], p[2]) and pair_key(*p) not in used
        ]
        # 优先选择也能帮助其他不足 state 的配对
        candidates.sort(
            key=lambda p: (
                max(0, min_per_state - appearances[p[0]]) +
                max(0, min_per_state - appearances[p[2]])
            ),
            reverse=True,
        )
        for p in candidates[:deficit]:
            add_pair(p)

    # 阶段 4：补齐跨 state 对到目标数量
    cross_count = sum(1 for p in selected if p[0] != p[2])
    remaining_cross = [p for p in all_cross if pair_key(*p) not in used]
    need_cross = cross_target - cross_count
    if need_cross > 0:
        if len(remaining_cross) < need_cross:
            raise ValueError(f"跨 state 配对不足，无法填充到 {cross_target}")
        for p in random.sample(remaining_cross, need_cross):
            add_pair(p)

    # 最终校验
    if len(selected) != total_pairs:
        raise ValueError(f"最终对数 {len(selected)} 不等于目标 {total_pairs}")

    under_represented = [s for s in state_list if appearances[s] < min_per_state]
    if under_represented:
        raise ValueError(
            f"以下 state 在最终配对中出现次数不足 {min_per_state}：{under_represented}"
        )

    # 打乱顺序后创建输出
    random.shuffle(selected)

    rows = []
    for idx, (s1, v1, s2, v2) in enumerate(selected, start=1):
        pair_dir = record_dir / f"pair_{idx:03d}"
        pair_dir.mkdir(exist_ok=True)

        dest_a = pair_dir / f"video_A{v1.suffix}"
        dest_b = pair_dir / f"video_B{v2.suffix}"

        shutil.copy2(v1, dest_a)
        shutil.copy2(v2, dest_b)

        rows.append({
            "pair_id": idx,
            "folder": pair_dir.name,
            "same_state": 1 if s1 == s2 else 0,
            "video_A_state": s1,
            "video_A_original": v1.name,
            "video_B_state": s2,
            "video_B_original": v2.name,
        })

    # 写入 CSV
    csv_path = record_dir / "pair_records.csv"
    fieldnames = [
        "pair_id", "folder", "same_state",
        "video_A_state", "video_A_original",
        "video_B_state", "video_B_original",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 统计并输出
    same_final = sum(1 for r in rows if r["same_state"] == 1)
    cross_final = len(rows) - same_final
    print(f"完成。共生成 {len(rows)} 对视频：")
    print(f"  同 state 对：{same_final}")
    print(f"  跨 state 对：{cross_final}")
    print(f"  输出目录：{record_dir}")
    print(f"  CSV 文件：{csv_path}")
    print("\n每个 state 出现次数：")
    for state in state_list:
        print(f"    {state}: {appearances[state]} 次")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机抽取视频对用于人工类别判断")
    parser.add_argument("--base-dir", required=True, help="视频根目录，例如 .../A1_C1_C1-1c")
    parser.add_argument("--ext", default=VIDEO_EXT, help="视频文件扩展名（默认 .mp4）")
    parser.add_argument("--total", type=int, default=TOTAL_PAIRS, help="总对数（默认 200）")
    parser.add_argument("--same-ratio", type=float, default=SAME_STATE_RATIO,
                        help="同 state 对比例（默认 0.5）")
    parser.add_argument("--min-per-state", type=int, default=MIN_PAIRS_PER_STATE,
                        help="每个 state 至少出现次数（默认 3）")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 42）")
    args = parser.parse_args()

    create_video_pairs(
        base_dir=args.base_dir,
        ext=args.ext,
        total_pairs=args.total,
        same_state_ratio=args.same_ratio,
        min_per_state=args.min_per_state,
        seed=args.seed,
    )

