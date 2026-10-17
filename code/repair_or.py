"""Drop rows written before failed-call tagging existed, so a resume retries them.

A row with an empty answer AND no usage block never got a reply from the model: it is a
rate-limited call that exhausted its retries, not a non-compliant response. Those rows predate
the `failed` field and would otherwise read as done. Rows with a real reply are kept.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402

OUT = os.path.join(S.DATA, "scores_or.jsonl")


def main() -> None:
    rows = []
    with open(OUT, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    # Keep a row only if the model actually answered, or answered emptily WITH a usage block
    # (a real completion that happened to be empty). A row with neither is an upstream failure:
    # HTTP 200, no choices, no usage. Those go back in the queue.
    keep = [r for r in rows
            if not r.get("failed") and (r.get("raw", "").strip() or r.get("usage"))]
    dropped = len(rows) - len(keep)
    for r in keep:
        r.setdefault("failed", False)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    print(f"kept {len(keep)}, dropped {dropped} failed-call rows for retry")


if __name__ == "__main__":
    main()
