"""CG height over roof height for the passenger cars of NHTSA's inertial
parameter database (Heydinger et al. 1999, SAE 1999-01-1336), driver only,
no ballast; and the lowest-roof cars.  Parses the printed listing."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO = _os.path.normpath(_os.path.join(HERE, '..', '..', '..'))
import re
import numpy as np
import pypdf

r = pypdf.PdfReader(_os.path.join(HERE, "nhtsa1999.pdf"))
CARS = {"SW", "2S", "4S", "3H", "5H", "4W", "2C", "3C", "5W", "2H", "4H"}
row_a = re.compile(r"^(\d{4}) (.+?) (\S+) (\S+) (VN|PU|MP|SW|\d[A-Z]) +(\S+) +(\S+(?: \S+)?) ([FR4]) "
                   r"(\d\.\d{3}) (\d\.\d{3}) (\d\.\d{3}) (\S+) (\d{4,5})")
row_b_id = re.compile(r"^(\d{4}) (.+?) (\S+) (\S+) (VN|PU|MP|SW|\d[A-Z]) +(\S+) +(\S+(?: \S+)?) ([FR4])$")
num_b = re.compile(r"^(\d\.\d{3}) (\S+) (\S+) (\S+) (\S+)(?: (-?\d+))? (\S+) (\S+)$")
A, B = {}, {}
for page in r.pages:
    lines = [l.strip() for l in (page.extract_text() or "").splitlines()]
    if any("Wheel-" in l for l in lines[:4]):
        for l in lines:
            m = row_a.match(l)
            if m:
                g = m.groups()
                A.setdefault(g[2], []).append(dict(desc="%s %s" % (g[0], g[1]), type=g[4], occ=g[5],
                                                   ballast=g[6], drive=g[7], roof=g[11], weight=float(g[12])))
    elif any("CG Location" in l for l in lines):
        ids = [m.groups() for m in (row_b_id.match(l) for l in lines) if m]
        nums = [m.groups() for m in (num_b.match(l) for l in lines) if m]
        if len(ids) != len(nums):
            continue
        for g, n in zip(ids, nums):
            B.setdefault(g[2], []).append(dict(type=g[4], occ=g[5], ballast=g[6], h=n[1]))
rows = []
for vid, bl in B.items():
    for b in bl:
        a = next((a for a in A.get(vid, []) if a["occ"] == b["occ"] and a["ballast"] == b["ballast"]), None)
        if a is None or b["type"] not in CARS or b["h"] == "N/A" or b["occ"] != "1" or b["ballast"] != "0":
            continue
        try:
            roof = float(a["roof"])
        except ValueError:
            continue
        rows.append((a["desc"], float(b["h"]), roof, float(b["h"]) / roof, a["weight"] / 9.81, a["drive"]))
rows = sorted(set(rows))
q = np.array([x[3] for x in rows])
print("passenger cars, driver only, no ballast: %d" % len(rows))
print("CG / roof height: mean %.3f  median %.3f  sd %.3f  p10-p90 %.3f-%.3f  min %.3f  max %.3f"
      % (q.mean(), np.median(q), q.std(), *np.percentile(q, [10, 90]), q.min(), q.max()))
low = sorted(rows, key=lambda x: x[2])[:10]
ql = np.array([x[3] for x in low])
print("lowest 10 roofs: %d cars, ratio mean %.3f median %.3f min %.3f max %.3f" % (len(low), ql.mean(), np.median(ql), ql.min(), ql.max()))
for d, h, roof, rr, m, dr in sorted(low, key=lambda x: x[2]):
    print("   %-38s roof %.2f m  CG %.3f m  ratio %.3f  %4.0f kg  drive %s" % (d, roof, h, rr, m, dr))
