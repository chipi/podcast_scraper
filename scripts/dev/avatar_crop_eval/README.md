# Avatar crop position — measurement loop

How `ProfileAvatar`'s `object-position: 50% 10%` was chosen (2026-09-30), and how to re-check it.
Person photos are portraits (taller than wide); a square avatar crop cuts them vertically, and a
centred crop took the top of the head off. This measures where to anchor instead.

macOS only (the detector uses Apple's Vision framework — no new dependency). Photos are fetched from
the public, unauthenticated photo route and kept out of the repo.

```bash
cd scripts/dev/avatar_crop_eval
mkdir -p img
# 1. sample: keep every name in names.txt that prod hosts a photo for
xargs -P 12 -I{} sh -c 'c=$(curl -s -m 15 -o img/{}.jpg -w "%{http_code}" \
  "https://closelistening.app/api/app/persons/person%3A{}/photo"); [ "$c" = 200 ] || rm -f img/{}.jpg' < names.txt
# 2. detect the largest face + eye line in each photo
swiftc -O detect.swift -o detect && ./detect img > faces.tsv
# 3. sweep the vertical anchor: heads cut (face top + 40% of face height for hair) and eye height
python3 sim.py
# 4. look at it: circles at 50 / 25 / 15 / 10 / 5 % for a set of photos (needs Pillow)
python3 sheet.py sheet.png barack-obama steven-sinofsky vladimir-putin
```

Result on 45 photos (43 portrait), 2026-09-30:

| anchor | heads cut | median eye height in the circle |
| ------ | --------- | ------------------------------- |
| 50% (centred, before) | 35 | 0.27 |
| 25% | 11 | 0.35 |
| 15% | 6 | 0.38 |
| **10% (chosen)** | **4** | **0.40** |
| 5% | 1 | 0.41 |

At 10% three of the four cuts are under 8px; the fourth (`steven-sinofsky`) is a source photo whose
face touches the top edge, which no anchor can frame. The hair allowance is an estimate — at 30% or
50% of face height the centred crop still cuts 26 or 40 of 43, so the direction does not depend on it.
