import sys

from PIL import Image, ImageDraw, ImageFont

POS = [0.50, 0.25, 0.15, 0.10, 0.05]
S = 110
PAD = 10
LABEL = 190
names = sys.argv[2:]
out = sys.argv[1]
W = LABEL + len(POS) * (S + PAD) + PAD
H = 40 + len(names) * (S + PAD)
sheet = Image.new("RGB", (W, H), (11, 14, 20))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except Exception:
    font = None
for j, p in enumerate(POS):
    d.text(
        (LABEL + j * (S + PAD) + PAD + 30, 12), f"{int(p*100)}%", fill=(230, 230, 230), font=font
    )
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=255)
for i, n in enumerate(names):
    y = 40 + i * (S + PAD)
    im = Image.open(f"img/{n}.jpg").convert("RGB")
    w, h = im.size
    d.text((PAD, y + S // 2 - 8), n.replace(".jpg", ""), fill=(200, 200, 200), font=font)
    for j, p in enumerate(POS):
        s = min(w, h)
        if h > w:
            box = (0, int((h - s) * p), s, int((h - s) * p) + s)  # object-position: 50% p
        else:
            box = (int((w - s) * 0.5), 0, int((w - s) * 0.5) + s, s)
        c = im.crop(box).resize((S, S))
        sheet.paste(c, (LABEL + j * (S + PAD) + PAD, y), mask)
sheet.save(out)
print(out, sheet.size)
