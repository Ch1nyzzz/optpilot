"""Exp 5: Offline vs Online comparison.

Compares offline Judge predictions against online ground truth
to quantify the gap between counterfactual evaluation and real execution.

Usage:
    python -m experiments.exp5_offline_vs_online --group B
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import LIBRARY_DIR, RESULTS_DIR
from optpilot.library.repair_library import RepairLibrary


def compare_offline_online(group_id: str = "B"):
    """Compare offline predictions with online results."""
    group_id = group_id.upper()
    offline_lib = RepairLibrary(LIBRARY_DIR / "offline_library.json")
    online_lib = RepairLibrary(LIBRARY_DIR / "online_library.json")

    offline_entries = [e for e in offline_lib.entries if e.fm_id == group_id]
    online_entries = [e for e in online_lib.entries if e.fm_id == group_id]

    print(f"=== Exp 5: Offline vs Online for Group-{group_id} ===")
    print(f"Offline entries: {len(offline_entries)}")
    print(f"Online entries: {len(online_entries)}")

    if not offline_entries or not online_entries:
        print("Need both offline and online results to compare. Run both pipelines first.")
        return

    # Match by repair description (approximate)
    offline_positive = sum(1 for e in offline_entries if e.n_success > 0)
    offline_negative = len(offline_entries) - offline_positive
    online_validated = sum(1 for e in online_entries if e.status == "validated")
    online_failed = sum(1 for e in online_entries if e.status == "failed")

    print(f"\nOffline Judge predictions:")
    print(f"  Would fix: {offline_positive}/{len(offline_entries)} ({offline_positive/len(offline_entries)*100:.1f}%)")
    print(f"  Would not fix: {offline_negative}/{len(offline_entries)} ({offline_negative/len(offline_entries)*100:.1f}%)")

    print(f"\nOnline ground truth:")
    print(f"  Validated: {online_validated}/{len(online_entries)} ({online_validated/len(online_entries)*100:.1f}%)")
    print(f"  Failed: {online_failed}/{len(online_entries)} ({online_failed/len(online_entries)*100:.1f}%)")

    # Compute agreement if we have matching entries
    print(f"\nAgreement analysis:")
    print(f"  Offline positive rate: {offline_positive/max(len(offline_entries),1)*100:.1f}%")
    print(f"  Online validation rate: {online_validated/max(len(online_entries),1)*100:.1f}%")

    results = {
        "fm_id": group_id,
        "offline_total": len(offline_entries),
        "offline_positive": offline_positive,
        "online_total": len(online_entries),
        "online_validated": online_validated,
    }
    out_path = RESULTS_DIR / f"exp5_offline_vs_online_group_{group_id.lower()}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="B")
    args = parser.parse_args()
    compare_offline_online(args.group)
