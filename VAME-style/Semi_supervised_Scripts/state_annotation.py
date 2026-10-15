"""读取 state 标注 CSV，生成 state → primitive 映射。"""

import argparse
import csv
import json
from pathlib import Path

from utils import load_config, output_dir, latest_experiment_dir


VALID_ACTIONS = {"keep", "merge", "delete", "unclear"}


def load_annotation_csv(csv_path):
    """读取标注 CSV，返回 list[dict]。"""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_mapping(rows):
    """从 CSV 行构建 state → primitive 映射。

    Returns
    -------
    mapping : dict[str, str]
        state_id -> primitive_name
    merge_groups : dict[str, list[int]]
        primitive -> list of state ids
    behavior_classes : list[str]
        所有 keep/merge 得到的 primitive 类别
    """
    mapping = {}
    merge_groups = {}
    deleted = []
    unclear = []

    for row in rows:
        state_id = row["state_id"].strip()
        action = row["action"].strip().lower()
        primitive = row.get("primitive_name", "").strip()
        merge_to = row.get("merge_to_state", "").strip()

        if action not in VALID_ACTIONS:
            raise ValueError(f"state {state_id}: 未知 action '{action}'")

        if action == "keep":
            if not primitive:
                raise ValueError(f"state {state_id}: action=keep 时必须填写 primitive_name")
            mapping[state_id] = primitive
            merge_groups.setdefault(primitive, []).append(int(state_id))

        elif action == "merge":
            if not merge_to:
                raise ValueError(f"state {state_id}: action=merge 时必须填写 merge_to_state")
            # 找到目标 state 的 primitive
            target_primitive = None
            for r in rows:
                if r["state_id"].strip() == merge_to and r["action"].strip().lower() in ("keep", "merge"):
                    target_primitive = r.get("primitive_name", "").strip()
                    if not target_primitive and r["action"].strip().lower() == "merge":
                        # 目标也是 merge，递归找最终 keep
                        target_primitive = mapping.get(merge_to)
                    break
            if not target_primitive:
                raise ValueError(f"state {state_id}: merge_to_state={merge_to} 没有对应 primitive")
            mapping[state_id] = target_primitive
            merge_groups.setdefault(target_primitive, []).append(int(state_id))

        elif action == "delete":
            deleted.append(int(state_id))

        elif action == "unclear":
            unclear.append(int(state_id))

    behavior_classes = sorted(merge_groups.keys())

    return mapping, merge_groups, behavior_classes, deleted, unclear


def save_mapping(out_path, mapping, merge_groups, behavior_classes, deleted, unclear):
    """保存映射为 JSON。"""
    data = {
        "behavior_classes": behavior_classes,
        "state_to_primitive": mapping,
        "merge_groups": {k: v for k, v in merge_groups.items()},
        "deleted_states": deleted,
        "unclear_states": unclear,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def print_summary(data):
    """打印映射摘要。"""
    print("=" * 50)
    print("State → Primitive 映射摘要")
    print("=" * 50)
    print(f"Primitive 类别数: {len(data['behavior_classes'])}")
    print(f"类别: {data['behavior_classes']}")
    print(f"已映射 state 数: {len(data['state_to_primitive'])}")
    print(f"删除 state 数: {len(data['deleted_states'])}")
    print(f"未确定 state 数: {len(data['unclear_states'])}")
    print("")

    print("每个 primitive 包含的 state:")
    for prim, states in data["merge_groups"].items():
        print(f"  {prim}: {states}")

    print("\n每个 state 的映射:")
    for sid in sorted(data["state_to_primitive"].keys(), key=lambda x: int(x)):
        print(f"  state {sid:2s} -> {data['state_to_primitive'][sid]}")


def main():
    parser = argparse.ArgumentParser(description="读取 state 标注 CSV 生成映射")
    parser.add_argument("--csv", type=str, required=True,
                        help="标注 CSV 路径")
    parser.add_argument("--out", type=str, default=None,
                        help="输出 JSON 路径，默认与 CSV 同名")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV: {csv_path}")

    rows = load_annotation_csv(csv_path)
    print(f"读取 {len(rows)} 行标注")

    mapping, merge_groups, behavior_classes, deleted, unclear = build_mapping(rows)

    out_path = Path(args.out) if args.out else csv_path.with_suffix(".json")
    data = save_mapping(out_path, mapping, merge_groups, behavior_classes, deleted, unclear)
    print(f"\n映射已保存: {out_path}")

    print_summary(data)


if __name__ == "__main__":
    main()