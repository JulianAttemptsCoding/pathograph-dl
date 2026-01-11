from __future__ import annotations

import re
from pathlib import Path

FILE = Path("pathograph/trade/faostat_step2.py")

def _patch_apply_lag_block(src: str) -> str:
    # Find the function by name
    m_start = re.search(r"(?m)^def\s+apply_lag_and_generate_pseudoflows\(", src)
    if not m_start:
        raise SystemExit("ERROR: Could not find function 'apply_lag_and_generate_pseudoflows' in faostat_step2.py")
    start_idx = m_start.start()

    # Find the end of this function (next top-level def or end of file)
    m_end = re.search(r"(?m)^def\s+", src[m_start.end():])
    if m_end:
        end_idx = m_start.end() + m_end.start()
    else:
        end_idx = len(src)

    head = src[:start_idx]
    block = src[start_idx:end_idx]
    tail = src[end_idx:]

    # Remove ANY existing months_processed increments inside the block
    # We use a non-greedy regex to match the lines
    block = re.sub(
        r"(?m)^[ \t]*stats\[['\"]months_processed['\"]\]\s*\+=\s*1\s*\n",
        "",
        block,
    )

    # Insert unconditional increment immediately after: for t in range(T):
    loop_pat = re.compile(r"(?m)^(?P<indent>[ \t]*)for\s+t\s+in\s+range\(T\):\s*$")
    m_loop = loop_pat.search(block)
    if not m_loop:
        raise SystemExit("ERROR: Could not find 'for t in range(T):' inside apply_lag_and_generate_pseudoflows")

    indent = m_loop.group("indent")
    insert_line = indent + "    " + 'stats["months_processed"] += 1'

    insert_pos = m_loop.end()
    block = block[:insert_pos] + "\n" + insert_line + block[insert_pos:]

    # Sanity: ensure exactly one increment remains in the block
    n_inc = len(re.findall(r"stats\[['\"]months_processed['\"]\]\s*\+=\s*1", block))
    if n_inc != 1:
        raise SystemExit(f"ERROR: Expected exactly 1 months_processed increment in block; found {n_inc}")

    return head + block + tail


def main() -> None:
    s = FILE.read_text(encoding="utf-8")

    # 1) Replace unicode arrows globally (Windows cp1252 console-safe)
    s = s.replace("\u2192", "->")

    # 2) Fix months_processed accounting in apply_lag_and_generate_pseudoflows
    s = _patch_apply_lag_block(s)

    # 3) Final sanity checks
    if "\u2192" in s:
        raise SystemExit("ERROR: Unicode arrow still present after replacement")

    FILE.write_text(s, encoding="utf-8")
    print(f"OK: patched {FILE}")


if __name__ == "__main__":
    main()
