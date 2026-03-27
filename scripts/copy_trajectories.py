#!/usr/bin/env python3
"""逐条复制 trajectory 到剪贴板，按回车继续下一条。"""

import json
import subprocess
import sys

JSONL_PATH = "data/annotations/ag2_6group_eval_100_blind.jsonl"

def copy_to_clipboard(text: str):
    subprocess.run(["pbcopy"], input=text.encode(), check=True)

def main():
    with open(JSONL_PATH) as f:
        lines = f.readlines()

    total = len(lines)
    for i, line in enumerate(lines, 1):
        record = json.loads(line)
        sid = record["sample_id"]
        traj = record["trajectory"]

        copy_to_clipboard(traj)
        print(f"[{i}/{total}] {sid} — 已复制到剪贴板 ({len(traj)} chars)")

        if i < total:
            try:
                input("按回车复制下一条 (Ctrl+C 退出)...")
            except KeyboardInterrupt:
                print("\n已退出。")
                sys.exit(0)

    print("全部复制完毕。")

if __name__ == "__main__":
    main()
