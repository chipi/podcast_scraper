import csv

rows = []
for r in csv.reader(open("faces.tsv"), delimiter="\t"):
    f, w, h, l, t, rt, b, eye, n = r[0], int(r[1]), int(r[2]), *map(float, r[3:8]), int(r[8])
    rows.append(dict(f=f, w=w, h=h, l=l, t=t, r=rt, b=b, eye=eye, n=n))
HAIR = 0.40  # head top ≈ face-box top minus 40% of face height


def evaluate(p, hair=HAIR):
    cut_top = cut_chin = 0
    eyes = []
    for x in rows:
        if x["h"] <= x["w"]:
            continue  # landscape/square: vertical fully visible
        win = x["w"] / x["h"]
        wt = (1 - win) * p
        wb = wt + win
        fh = x["b"] - x["t"]
        head = x["t"] - hair * fh
        if head < wt - 1e-9:
            cut_top += 1
        if x["b"] > wb + 1e-9:
            cut_chin += 1
        eyes.append((x["eye"] - wt) / win)
    eyes.sort()
    med = eyes[len(eyes) // 2] if eyes else 0
    return cut_top, cut_chin, med, len(eyes)


portrait = sum(1 for x in rows if x["h"] > x["w"])
print(f"{len(rows)} photos, {portrait} taller than wide (only those get a vertical crop)")
print(" pos%  head-top-cut  chin-cut  median-eye-height-in-circle")
for p in [0, 0.05, 0.10, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40, 0.50]:
    ct, cc, med, n = evaluate(p)
    print(f"{p*100:5.1f}  {ct:12d}  {cc:8d}  {med:.2f}")
print("\nsensitivity to the hair estimate (head-top cuts):")
for hair in (0.3, 0.4, 0.5):
    print(
        f" hair={hair}: "
        + "  ".join(
            f"{int(p*100)}%->{evaluate(p,hair)[0]}" for p in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        )
    )
print("\ncut at the CURRENT 50% (head top), per photo:")
for x in rows:
    if x["h"] <= x["w"]:
        continue
    win = x["w"] / x["h"]
    wt = (1 - win) * 0.5
    head = x["t"] - HAIR * (x["b"] - x["t"])
    if head < wt:
        print(
            f"  {x['f']:30s} head top {head:.3f} vs window top {wt:.3f}"
            f"  (aspect {x['h']/x['w']:.2f})"
        )
