"""Phase 1.1 - parse the PROVA 200A CSV exports."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse(path):
    """{sample_no: {"meta": {dt, voc, isc}, "d": [(V, I, P), ...]}} - one entry per block."""
    samples, cur = {}, None
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        s = [x.strip() for x in line.strip().split(",")]
        if not s or s == [""]:
            continue
        if s[0].startswith("Sample No."):
            cur = int(s[1])
            samples[cur] = {"meta": {}, "d": []}
        elif cur is None:
            continue                                   # "Sample Count" line
        elif s[0].startswith("Date"):
            samples[cur]["meta"]["dt"] = s[1]
        elif s[0].startswith("Vopen"):
            samples[cur]["meta"]["voc"] = float(s[1])
        elif s[0].startswith("Ishort"):
            samples[cur]["meta"]["isc"] = float(s[1])
        elif s[0].startswith("V (V)"):
            pass
        else:
            samples[cur]["d"].append(tuple(map(float, s[:3])))
    return samples


def load_all(cfg=None):
    """{file_tag: {sample: {"meta", "V", "I", "P"}}} for both files."""
    cfg = cfg or load_config()
    out = {}
    for tag, rel in cfg["files"].items():
        out[tag] = {}
        for n, s in parse(ROOT / rel).items():
            V, I, P = np.array(s["d"], float).T
            out[tag][n] = {"meta": s["meta"], "V": V, "I": I, "P": P}
    return out


if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    for tag, S in data.items():
        sizes = {len(s["V"]) for s in S.values()}
        sel = [n for pair in cfg["panels"][tag] for n in pair]
        missing = [n for n in sel if n not in S]
        print(f"{tag}: {len(S)} samples, points per sample {sizes}, selected {len(sel)}, missing {missing}")
