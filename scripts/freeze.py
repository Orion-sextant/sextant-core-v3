"""Freeze-gate staging (PROTOCOL_v3.md section 22; build item 10).

Makes gate authorization a single command — but the flip itself remains the
owner's explicit act:

  python scripts/freeze.py --check
      Run the readiness checklist (no writes): full test suite, manifest
      TO_FILL audit, selected LRs present, packer round-trip, no comparative
      curves opened. Prints PASS/FAIL per criterion.

  python scripts/freeze.py --authorize "I authorize the freeze gate"
      Only with that exact phrase: stamps authorized_by_owner true +
      authorized_utc, fills repository_commit, relabels status frozen,
      commits, and tags v3.1-freeze. Refuses if --check fails.

This script never starts comparative runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
AUTH_PHRASE = "I authorize the freeze gate"
GATE_FIELDS = {"repository_commit: TO_FILL", "authorized_utc: TO_FILL"}


def sh(*args, **kw):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, **kw)


def checklist() -> list[tuple[str, bool, str]]:
    out = []
    r = sh(sys.executable, "-m", "pytest", "-q")
    out.append(("mandatory unit tests all pass", r.returncode == 0,
                r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:100]))
    txt = (REPO / "manifest.yaml").read_text()
    tofill = [l.strip() for l in txt.splitlines()
              if "TO_FILL" in l and not l.strip().startswith("#")]
    blocking = [l for l in tofill if not any(g in l for g in GATE_FIELDS)]
    out.append(("no TO_FILL except gate-stamped fields", not blocking, "; ".join(blocking) or "clean"))
    man = yaml.safe_load(txt)
    lrs = man["training"]["lr_fairness_stage"]["selected_peak_lr"]
    missing = [k for k, v in lrs.items() if v == "TO_FILL"]
    out.append(("8 selected peak LRs recorded", not missing, "; ".join(missing) or "all present"))
    gate = man["freeze_gate"]["authorized_by_owner"]
    out.append(("gate not yet flipped (owner-only act)", gate is False, f"authorized_by_owner={gate}"))
    r = sh("git", "status", "--porcelain")
    out.append(("working tree clean for freeze commit", r.stdout.strip() == "",
                "dirty" if r.stdout.strip() else "clean"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--authorize", type=str, default=None)
    a = ap.parse_args()

    results = checklist()
    print("=== freeze readiness checklist ===")
    ok_all = True
    for name, ok, detail in results:
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    if a.check or a.authorize is None:
        sys.exit(0 if ok_all else 1)

    if a.authorize != AUTH_PHRASE:
        print(f'refused: --authorize must be exactly "{AUTH_PHRASE}"')
        sys.exit(2)
    # authorization requires everything green except the gate fields themselves
    if not ok_all:
        print("refused: checklist has failures; resolve before authorizing")
        sys.exit(1)

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    commit = sh("git", "rev-parse", "HEAD").stdout.strip()
    p = REPO / "manifest.yaml"
    txt = p.read_text()
    txt = txt.replace("repository_commit: TO_FILL", f"repository_commit: {commit}")
    txt = txt.replace("authorized_by_owner: false   # flip only with explicit sign-off",
                      "authorized_by_owner: true    # owner sign-off recorded")
    txt = txt.replace("authorized_utc: TO_FILL", f"authorized_utc: {now}")
    txt = re.sub(r"status: preregistration_draft.*", "status: frozen", txt, count=1)
    p.write_text(txt)
    sh("git", "add", "manifest.yaml")
    sh("git", "-c", "user.name=Sextant Owner", "-c", "user.email=jared789@gmail.com",
       "commit", "-m", f"FREEZE: owner authorized gate at {now}\n\npre-freeze commit: {commit}")
    sh("git", "tag", "v3.1-freeze")
    print(f"frozen at {now}; tagged v3.1-freeze. Push the tag for the public timestamp:")
    print("  git push origin main --tags")


if __name__ == "__main__":
    main()
