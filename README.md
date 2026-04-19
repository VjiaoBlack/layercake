# layercake

Cut a photo into depth-ordered transparent PNG layers with [Meta's
Segment Anything 2](https://github.com/facebookresearch/sam2). Every output
PNG is the source image's exact dimensions, so layers stack cleanly under
CSS `object-fit: cover; object-position: center`.

PSD is for Photoshop. **layercake is for CSS.**

Built for layered hero composites on the web — the kind where text weaves
between a foreground object and the subject, or where parallax layers
separate fore/mid/background. Click a few points per layer, optionally draw
a bounding box, pick edge quality, save. Source dimensions guaranteed.

## Why this and not something else

| Tool | Click-prompted | N layers | Source-dim PNG stack | Local & free |
|---|:-:|:-:|:-:|:-:|
| Photoshop "Select Subject" | ✕ | ✕ | N/A | Paid |
| remove.bg / Photoroom / Clipdrop | ✕ | Fg/bg only | ✕ | Cloud |
| iOS "Lift Subject" | ✕ | Subject only | ✕ | Local |
| [jhj0517/sam2-playground](https://github.com/jhj0517/sam2-playground) | ✓ | PSD | ✕ | Local |
| 10b.ai RGBA Layers | ✕ | ✓ | ✓ | Cloud, paid |
| **layercake** | **✓** | **✓** | **✓** | **✓** |

## Install

Requires Python 3.10+. Recommended via [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/<you>/layercake.git
cd layercake
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

First run downloads SAM 2 weights (~900 MB for `sam2-hiera-large`). Cached
under `~/.cache/huggingface/`. Device auto-detects MPS (Apple silicon),
CUDA, or CPU.

## Interactive UI (recommended)

```bash
python layers_app.py
```

Opens at `http://127.0.0.1:7860`. Workflow:

1. Upload a source image.
2. Name a layer (order = depth; **first layer = nearest**) and hit **Add layer**.
3. With the layer active, click the image. Modes:
   - **include** — positive click: "this pixel belongs to the layer."
   - **exclude** — negative click: "this pixel does NOT."
   - **move** — 1st click picks up the nearest point, 2nd drops it.
   - **box** — 2 clicks mark opposite corners of an axis-aligned bounding box.
   - **erase** — click near a point (or inside the box) to remove it.
4. Repeat for each layer. Edit points directly in the dataframe (move by
   changing x/y, delete by removing rows).
5. Hit **Save layers**.

Output directory gets:

- `<name>.png` per layer, all source-dim RGBA
- `bg.png` — the inverse of the union of all your layers
- `points.json` — the full spec, for replay via the CLI
- `snippet.html` — ready-to-paste HTML + CSS that stacks the layers

## Headless CLI

Same engine, no UI. Useful in build scripts.

```bash
python layers.py input.jpg \
  --layers '[
    {"name": "foreground",  "points": [[1200, 300], [850, 250]]},
    {"name": "subject",     "points": [[420, 400]], "box": [300, 200, 600, 900]},
    {"name": "midground",   "points": [[1100, 800]], "labels": [1]}
  ]' \
  --out out/ \
  --edges matting \
  --preview
```

Each layer entry supports:

- `name` (required)
- `points` — list of `[x, y]` in source-image pixel space
- `labels` — per-point include/exclude: `1` or `0` (default all 1)
- `box` — optional axis-aligned `[x1, y1, x2, y2]`

`--layers` also accepts a path to a JSON file (e.g., `points.json` exported
from the UI).

### Flags

| Flag | Default | What |
|---|---|---|
| `--model` | `sam2-hiera-large` | HF id `facebook/sam2-hiera-{large,base-plus,small,tiny}` |
| `--device` | `auto` | `auto`/`cpu`/`cuda`/`mps` |
| `--edges` | `feather` | `feather` (Gaussian blur), `sam-soft` (SAM's sigmoid logits), or `matting` (pymatting closed-form) |
| `--feather` | `2` | Blur radius (px) for `--edges feather` |
| `--matting-band` | `8` | Unknown-region band width (px) for `--edges matting` |
| `--preview` | off | Write a tinted `_preview.png` |
| `--no-bg` | off | Skip `bg.png` |
| `--no-depth` | off | Skip the Depth Anything V2 depth map |
| `--no-css` | off | Skip `snippet.html` |
| `--infill` | `none` | Fill the bg "hole" where layers sit: `none`, `opencv` (Navier-Stokes, instant), `lama` (LaMa model, ~200 MB, plausible on complex scenes) |

### Depth map (optional)

If you want depth-driven parallax speeds per layer, layercake can also emit
a grayscale depth map via [Depth Anything
V2](https://github.com/DepthAnything/Depth-Anything-V2). Runs alongside the
SAM 2 pass; adds a few seconds. On by default; skip with `--no-depth`.

## Edge quality: feather vs. matting

**feather** (default) — post-processes the hard mask with a Gaussian blur.
Fast, uniform, content-blind. Good for most cases; leaves halos on
high-contrast edges and doesn't catch thin filaments.

**matting** — [pymatting](https://pymatting.github.io/) alpha matting.
Builds a trimap (erode = known-fg, dilate-inverse = known-bg, thin band =
unknown), then solves for soft alpha in the unknown band using actual
image structure. Edge-aware; dramatically better on hair, fur, and fine
filaments.

`--matting-band` controls the unknown-zone width: narrow (2-4) for tight
boundaries, medium (8-12) for soft edges and hair, wide (16-24) for wispy
structures or motion blur.

`--matting-algo` picks the solver: `cf` (default, closed-form — fastest
and highest quality on typical problems), `lbdm` (approximation for very
large unknown regions), `knn` (alternative kernel, sometimes useful on
fur).

## Background infill

By default `bg.png` has a transparent "hole" where your foreground layers
sit — perfect for CSS stacking since the foreground layer sits on top and
fills the hole visually.

If you want the bg as a standalone complete image (e.g., for parallax where
the foreground slides and the exposed bg should look plausible), use
`--infill`:

- `--infill opencv` — OpenCV Navier-Stokes inpainting. Near-instant, zero
  extra deps beyond `opencv-python-headless`. Works well on simple
  backdrops (walls, gradients, out-of-focus surfaces).
- `--infill lama` — [LaMa](https://github.com/advimman/lama) model via
  `simple-lama-inpainting`. Plausible on complex scenes; first use
  downloads ~200 MB; inference is a few seconds.

Output `bg.png` becomes fully opaque (alpha=1 everywhere) and contains the
infilled backdrop.

## CSS integration

`snippet.html` ships ready to paste:

```html
<div class="hero">
  <img src="images/bg.png" class="layer l-bg" alt="">
  <h1 class="hero-wordmark">your wordmark</h1>
  <img src="images/subject.png" class="layer l-subject" alt="">
  <img src="images/foreground.png" class="layer l-foreground" alt="">
</div>
```

```css
.hero { position: relative; isolation: isolate; overflow: hidden; }
.hero .layer { position: absolute; inset: 0; width: 100%; height: 100%;
               object-fit: cover; object-position: center;
               pointer-events: none; }
.hero .l-bg { z-index: 1; }
.hero .hero-wordmark { z-index: 2; position: relative; /* your styles */ }
.hero .l-subject { z-index: 3; }
.hero .l-foreground { z-index: 4; }
```

The wordmark sits between the background and the subject — text weaves
through the composite naturally.

## Workflow tips

**Many small layers beats one big layer.** SAM 2 hits diminishing returns
after ~10-15 points per blob. Split contiguous regions into separate layers
(`hand-left`, `hand-right`, …) with 1-3 points each rather than 30 points
on one layer.

**Boxes for the easy 80%, points for the fiddly 20%.** A box around a whole
object is often enough; add positive points for missed bits and negative
points to pry neighbors apart.

**Click the things you care about; let `bg.png` catch the rest.** You don't
need to enumerate every region.

**Large model is worth it.** The image encoder runs once per upload; all
clicks after that are cheap. `sam2-hiera-large` materially improves
boundaries on thin structures vs. smaller variants.

## Credits

Built on top of:

- [Segment Anything 2](https://github.com/facebookresearch/sam2) (Meta)
- [pymatting](https://pymatting.github.io/) (closed-form alpha matting)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Gradio](https://gradio.app) for the UI

## License

MIT — see [LICENSE](LICENSE).
