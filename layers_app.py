#!/usr/bin/env python3
"""
layercake — interactive Gradio UI for the layers.py CLI.

Workflow:
  1. Upload source image.
  2. Add named layers (order = depth; first = nearest).
  3. For the active layer, click to add points (include/exclude), drop a
     box with two clicks, move a point (two clicks: pick + drop), or erase.
  4. Save → depth-ordered RGBA PNGs at source dims + points.json + snippet.html.

Notes:
  - SAM 2 supports multiple points + one axis-aligned box per predict() call.
    Multiple contiguous blobs? Make multiple layers rather than cramming one.
  - Undo pops the last added thing (point or box) for the active layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from layers import (
    _coerce_alpha,
    autocast_ctx,
    build_css_snippet,
    detect_device,
    load_sam2_predictor,
    mask_to_rgba,
    matting_refine,
)

LAYER_COLORS = [
    (255, 80, 80),
    (80, 200, 120),
    (80, 140, 255),
    (230, 190, 80),
    (200, 100, 230),
    (80, 220, 220),
]

PREVIEW_MAX_DIM = 1280  # longest edge of the click canvas; source is always full-res

STATE: dict = {
    "predictor": None,
    "model_loaded": None,
    "device_resolved": None,
    "rgb": None,
    "preview_scale": 1.0,         # preview_dim / source_dim; clicks are in preview coords
    "cached_masks": {},
    "pending_box_corner": None,   # [x, y] in SOURCE coords, awaiting 2nd box click
    "move_selected": None,        # (layer_name, point_idx) awaiting drop
}


def _load_predictor(model: str, device: str):
    device_resolved = detect_device(device)
    if (STATE["predictor"] is None or STATE["model_loaded"] != model
            or STATE["device_resolved"] != device_resolved):
        STATE["predictor"] = load_sam2_predictor(model, device_resolved)
        STATE["model_loaded"] = model
        STATE["device_resolved"] = device_resolved
    return STATE["predictor"], device_resolved


def _new_layer(name: str) -> dict:
    return {"name": name, "points": [], "labels": [], "box": None,
            "history": []}  # history entries: ("point", idx) | ("box", prev_box)


def _segment(layer: dict) -> Optional[np.ndarray]:
    """Run SAM 2 on an active layer's prompts. None if nothing to segment."""
    if not layer["points"] and layer["box"] is None:
        return None
    predictor = STATE["predictor"]
    pts = np.asarray(layer["points"], dtype=np.float32) if layer["points"] else None
    lbls = np.asarray(layer["labels"], dtype=np.int32) if layer["points"] else None
    box = np.asarray(layer["box"], dtype=np.float32) if layer["box"] is not None else None
    with autocast_ctx(STATE["device_resolved"]):
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=lbls, box=box, multimask_output=True,
        )
    best = int(np.argmax(scores))
    return np.asarray(masks[best]).astype(bool)


def _render(layers_state: list[dict], active_idx: int) -> Optional[np.ndarray]:
    """Compose preview at full source dims then downscale to PREVIEW_MAX_DIM for speed."""
    rgb = STATE["rgb"]
    if rgb is None:
        return None
    H, W, _ = rgb.shape
    canvas = Image.fromarray(rgb).convert("RGBA")
    # Tint all layers, back→front (so nearer layer tints sit on top).
    for idx, layer in list(enumerate(layers_state))[::-1]:
        mask = STATE["cached_masks"].get(layer["name"])
        if mask is None:
            continue
        color = LAYER_COLORS[idx % len(LAYER_COLORS)]
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[..., :3] = color
        overlay[..., 3] = mask.astype(np.uint8) * 90
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, "RGBA"))
    # Overlay active layer's prompts.
    if 0 <= active_idx < len(layers_state):
        draw = ImageDraw.Draw(canvas)
        r = max(6, min(W, H) // 180)
        layer = layers_state[active_idx]
        if layer["box"] is not None:
            x1, y1, x2, y2 = layer["box"]
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 220, 60, 255), width=max(2, r // 2))
        if STATE["pending_box_corner"] is not None:
            x, y = STATE["pending_box_corner"]
            draw.ellipse((x - r, y - r, x + r, y + r),
                         fill=(255, 220, 60, 255), outline=(255, 255, 255, 255), width=2)
        sel = STATE["move_selected"]
        sel_idx = sel[1] if sel and sel[0] == layer["name"] else -1
        for i, ((x, y), lbl) in enumerate(zip(layer["points"], layer["labels"])):
            fill = (0, 230, 50, 255) if lbl == 1 else (240, 60, 60, 255)
            if i == sel_idx:
                # "Picked up" highlight — bigger ring in yellow.
                rr = int(r * 1.8)
                draw.ellipse((x - rr, y - rr, x + rr, y + rr),
                             outline=(255, 220, 60, 255), width=max(3, r // 2))
            draw.ellipse((x - r, y - r, x + r, y + r),
                         fill=fill, outline=(255, 255, 255, 255), width=2)
    # Downscale to preview size. Clicks will come back in preview coords and get
    # rescaled by STATE["preview_scale"] before segmentation.
    scale = STATE["preview_scale"]
    if scale < 1.0:
        pw, ph = int(round(W * scale)), int(round(H * scale))
        canvas = canvas.resize((pw, ph), Image.BILINEAR)
    return np.asarray(canvas.convert("RGB"))


def _compute_preview_scale(W: int, H: int) -> float:
    longest = max(W, H)
    if longest <= PREVIEW_MAX_DIM:
        return 1.0
    return PREVIEW_MAX_DIM / float(longest)


def _find_idx(layers_state: list[dict], name: Optional[str]) -> int:
    if not name:
        return -1
    for i, L in enumerate(layers_state):
        if L["name"] == name:
            return i
    return -1


def _nearest_point_idx(layer: dict, x: float, y: float, threshold: float) -> int:
    best, best_d = -1, threshold * threshold
    for i, (px, py) in enumerate(layer["points"]):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best, best_d = i, d
    return best


def _recompute(layer: dict) -> Optional[np.ndarray]:
    mask = _segment(layer)
    if mask is None:
        STATE["cached_masks"].pop(layer["name"], None)
    else:
        STATE["cached_masks"][layer["name"]] = mask
    return mask


def _layer_to_df(layer: Optional[dict]) -> list[list]:
    if layer is None:
        return []
    return [[int(x), int(y), "+" if lbl == 1 else "−"]
            for (x, y), lbl in zip(layer["points"], layer["labels"])]


def _df_to_layer_fields(df_rows) -> tuple[list[list[float]], list[int]]:
    """Parse dataframe rows into (points, labels). Tolerant of blanks/invalid."""
    if df_rows is None:
        return [], []
    # Gradio sends either a DataFrame-like with .values or a list of lists.
    rows = df_rows.values.tolist() if hasattr(df_rows, "values") else list(df_rows)
    points: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if row is None or len(row) < 3:
            continue
        try:
            x = float(row[0]); y = float(row[1])
        except (TypeError, ValueError):
            continue
        lbl_raw = str(row[2]).strip() if row[2] is not None else "+"
        lbl = 0 if lbl_raw in ("-", "−", "0", "exclude", "ex") else 1
        points.append([x, y])
        labels.append(lbl)
    return points, labels


def _prompt_summary(layer: dict) -> str:
    n_in = sum(1 for l in layer["labels"] if l == 1)
    n_ex = sum(1 for l in layer["labels"] if l == 0)
    parts = []
    if n_in: parts.append(f"{n_in}+")
    if n_ex: parts.append(f"{n_ex}−")
    if layer["box"] is not None: parts.append("1 box")
    return ", ".join(parts) if parts else "(empty)"


# --- handlers ---

def on_upload(image, model, device, _layers_state):
    if image is None:
        return gr.update(), [], gr.update(choices=[], value=None), "No image.", []
    rgb = np.asarray(image.convert("RGB"))
    predictor, device_resolved = _load_predictor(model, device)
    with autocast_ctx(device_resolved):
        predictor.set_image(rgb)
    STATE["rgb"] = rgb
    STATE["cached_masks"] = {}
    STATE["pending_box_corner"] = None
    STATE["move_selected"] = None
    H, W, _ = rgb.shape
    STATE["preview_scale"] = _compute_preview_scale(W, H)
    return (_render([], -1), [], gr.update(choices=[], value=None),
            f"Loaded {W}×{H} on {device_resolved} ({model}). "
            f"Preview @ {int(W*STATE['preview_scale'])}×{int(H*STATE['preview_scale'])}. "
            f"Add a layer to start.", [])


def on_add_layer(new_name, layers_state):
    new_name = (new_name or "").strip()
    if not new_name:
        return layers_state, gr.update(), gr.update(value=""), "Name can't be empty."
    if any(L["name"] == new_name for L in layers_state):
        return layers_state, gr.update(), gr.update(value=""), f"Layer {new_name!r} exists."
    layers_state = layers_state + [_new_layer(new_name)]
    choices = [L["name"] for L in layers_state]
    return (layers_state, gr.update(choices=choices, value=new_name),
            gr.update(value=""), f"Added layer {new_name!r} (depth #{len(layers_state)}).")


def on_remove_layer(active_layer_name, layers_state):
    layers_state = [L for L in layers_state if L["name"] != active_layer_name]
    STATE["cached_masks"].pop(active_layer_name, None)
    STATE["pending_box_corner"] = None
    choices = [L["name"] for L in layers_state]
    new_active = choices[0] if choices else None
    preview = _render(layers_state, 0 if choices else -1)
    new_layer = next((L for L in layers_state if L["name"] == new_active), None)
    return (layers_state, gr.update(choices=choices, value=new_active),
            preview, f"Removed {active_layer_name!r}.", _layer_to_df(new_layer))


def on_clear_points(active_layer_name, layers_state):
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer.", []
    layers_state[idx].update(points=[], labels=[], box=None, history=[])
    STATE["cached_masks"].pop(active_layer_name, None)
    STATE["pending_box_corner"] = None
    return layers_state, _render(layers_state, idx), f"Cleared {active_layer_name!r}.", []


def on_image_click(evt: gr.SelectData, mode, active_layer_name, layers_state):
    if STATE["rgb"] is None:
        return layers_state, None, "Upload an image first.", []
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "Add and select a layer first.", []
    layer = layers_state[idx]
    scale = STATE["preview_scale"]
    # evt.index is in preview coords; divide by scale to get source-image pixels.
    x = int(round(evt.index[0] / scale))
    y = int(round(evt.index[1] / scale))
    H, W, _ = STATE["rgb"].shape
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))

    if mode in ("include", "exclude"):
        lbl = 1 if mode == "include" else 0
        layer["points"].append([x, y])
        layer["labels"].append(lbl)
        layer["history"].append(("point", None))
        _recompute(layer)
        msg = f"{layer['name']}: {_prompt_summary(layer)}"
    elif mode == "box":
        if STATE["pending_box_corner"] is None:
            STATE["pending_box_corner"] = [x, y]
            msg = f"{layer['name']}: set first corner, click opposite corner."
        else:
            px, py = STATE["pending_box_corner"]
            STATE["pending_box_corner"] = None
            prev_box = layer["box"]
            layer["box"] = [float(min(px, x)), float(min(py, y)),
                            float(max(px, x)), float(max(py, y))]
            layer["history"].append(("box", prev_box))
            _recompute(layer)
            msg = f"{layer['name']}: box set — {_prompt_summary(layer)}"
    elif mode == "move":
        thresh = max(W, H) / 40.0
        sel = STATE["move_selected"]
        if sel is None or sel[0] != layer["name"]:
            # First click: pick up nearest point.
            ni = _nearest_point_idx(layer, x, y, thresh)
            if ni >= 0:
                STATE["move_selected"] = (layer["name"], ni)
                msg = f"{layer['name']}: picked point #{ni}. Click to drop."
            else:
                msg = "Move: no point within reach to pick up."
        else:
            # Second click: drop the selected point here.
            _, ni = sel
            if 0 <= ni < len(layer["points"]):
                layer["points"][ni] = [x, y]
                _recompute(layer)
            STATE["move_selected"] = None
            msg = f"{layer['name']}: moved — {_prompt_summary(layer)}"
    elif mode == "erase":
        thresh = max(W, H) / 60.0
        ni = _nearest_point_idx(layer, x, y, thresh)
        if ni >= 0:
            layer["points"].pop(ni)
            layer["labels"].pop(ni)
            layer["history"].append(("point-removed", ni))
            _recompute(layer)
            msg = f"{layer['name']}: point removed — {_prompt_summary(layer)}"
        elif layer["box"] is not None and _inside_box(layer["box"], x, y):
            layer["history"].append(("box-removed", layer["box"]))
            layer["box"] = None
            _recompute(layer)
            msg = f"{layer['name']}: box removed — {_prompt_summary(layer)}"
        else:
            msg = "Erase: no nearby point or box."
    else:
        msg = f"Unknown mode {mode!r}"
    return layers_state, _render(layers_state, idx), msg, _layer_to_df(layer)


def _inside_box(box: list[float], x: float, y: float) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def on_undo(active_layer_name, layers_state):
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer.", []
    layer = layers_state[idx]
    if not layer["history"]:
        return layers_state, _render(layers_state, idx), "Nothing to undo.", _layer_to_df(layer)
    kind, data = layer["history"].pop()
    if kind == "point":
        layer["points"].pop()
        layer["labels"].pop()
    elif kind == "box":
        layer["box"] = data
    elif kind == "box-removed":
        layer["box"] = data
    STATE["pending_box_corner"] = None
    _recompute(layer)
    return layers_state, _render(layers_state, idx), f"Undo — {_prompt_summary(layer)}", _layer_to_df(layer)


def on_active_layer_change(active_layer_name, layers_state):
    STATE["pending_box_corner"] = None
    STATE["move_selected"] = None
    idx = _find_idx(layers_state, active_layer_name)
    layer = layers_state[idx] if idx >= 0 else None
    return _render(layers_state, idx), _layer_to_df(layer)


def on_mode_change(_mode, active_layer_name, layers_state):
    """Clear pending interactions so a mode switch doesn't dangle state."""
    STATE["pending_box_corner"] = None
    STATE["move_selected"] = None
    idx = _find_idx(layers_state, active_layer_name)
    return _render(layers_state, idx)


def on_points_df_change(df_rows, active_layer_name, layers_state):
    """User edited the points dataframe: mutate the active layer and re-segment."""
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer."
    layer = layers_state[idx]
    points, labels = _df_to_layer_fields(df_rows)
    # Skip no-op changes to avoid feedback loops.
    if points == layer["points"] and labels == layer["labels"]:
        return layers_state, _render(layers_state, idx), gr.update()
    layer["points"] = points
    layer["labels"] = labels
    _recompute(layer)
    return layers_state, _render(layers_state, idx), f"{layer['name']}: {_prompt_summary(layer)}"


def on_save(out_dir, edges, feather, matting_band, include_bg, include_css, layers_state):
    if STATE["rgb"] is None:
        return "No image loaded.", ""
    if not layers_state:
        return "No layers to save.", ""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rgb = STATE["rgb"]
    H, W, _ = rgb.shape

    claimed = np.zeros((H, W), dtype=bool)
    exclusive = []
    for L in layers_state:
        m = STATE["cached_masks"].get(L["name"])
        if m is None:
            continue
        m_excl = m & ~claimed
        exclusive.append((L["name"], m_excl))
        claimed |= m

    finals = []
    for name, m_excl in exclusive:
        if edges == "matting":
            try:
                alpha = matting_refine(rgb, m_excl, band_px=int(matting_band))
            except ImportError:
                alpha = m_excl
        else:
            alpha = m_excl
        finals.append((name, alpha))

    written = []
    for name, a in finals:
        fe = int(feather) if edges == "feather" else 0
        rgba = mask_to_rgba(rgb, a, fe)
        p = out / f"{name}.png"
        rgba.save(p, optimize=True)
        written.append(p.name)

    if include_bg:
        union = np.zeros((H, W), dtype=np.float32)
        for _n, a in finals:
            union = 1.0 - (1.0 - union) * (1.0 - _coerce_alpha(a))
        bg_alpha = 1.0 - union
        fe = int(feather) if edges == "feather" else 0
        rgba = mask_to_rgba(rgb, bg_alpha, fe)
        p = out / "bg.png"
        rgba.save(p, optimize=True)
        written.append(p.name)

    (out / "points.json").write_text(json.dumps(
        [{"name": L["name"], "points": L["points"], "labels": L["labels"], "box": L["box"]}
         for L in layers_state],
        indent=2,
    ))

    css = ""
    if include_css:
        css = build_css_snippet([name for name, _ in finals], include_bg=include_bg)
        (out / "snippet.html").write_text(css)
        written.append("snippet.html")

    status = (f"Saved {len(written)} file(s) at {W}×{H} → `{out}`\n"
              f"Files: {', '.join(written)}\nSpec: points.json\nEdges: {edges}")
    return status, css


def on_export_spec(layers_state):
    return json.dumps(
        [{"name": L["name"], "points": L["points"], "labels": L["labels"], "box": L["box"]}
         for L in layers_state],
        indent=2,
    )


# --- UI ---

def build_ui(args):
    with gr.Blocks(title="layercake — SAM 2 interactive") as demo:
        gr.Markdown(
            "# layercake\n"
            "Upload → add layers (order = depth, first = nearest) → click points "
            "(green=include, red=exclude) or draw a box (two clicks) → Save.  \n"
            "**Tip:** for complex scenes, many small named layers beat cramming many "
            "points into one. Use Erase to remove the nearest point/box."
        )
        layers_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Source image", height=220)
                with gr.Row():
                    model = gr.Dropdown(
                        ["sam2-hiera-large", "sam2-hiera-base-plus",
                         "sam2-hiera-small", "sam2-hiera-tiny"],
                        value=args.model, label="Model",
                    )
                    device = gr.Dropdown(["auto", "cpu", "cuda", "mps"],
                                         value=args.device, label="Device")

                gr.Markdown("### Layers")
                with gr.Row():
                    new_name = gr.Textbox(label="New layer name", placeholder="subject", scale=2)
                    add_btn = gr.Button("Add layer", scale=1)
                active_layer = gr.Dropdown(choices=[], label="Active layer")
                point_mode = gr.Radio(
                    ["include", "exclude", "move", "box", "erase"],
                    value="include", label="Click mode",
                    info="include/exclude: add point. move: 1st click picks nearest, 2nd drops it. "
                         "box: two clicks = opposite corners. erase: click near a point/box to remove it.",
                )
                with gr.Row():
                    undo_btn = gr.Button("Undo last")
                    clear_btn = gr.Button("Clear layer")
                    remove_btn = gr.Button("Remove layer", variant="stop")

                points_df = gr.Dataframe(
                    headers=["x", "y", "±"],
                    datatype=["number", "number", "str"],
                    row_count=(0, "dynamic"),
                    column_count=(3, "fixed"),
                    label="Points (edit x/y to move, delete row to remove)",
                    interactive=True,
                )

                gr.Markdown("### Save")
                out_dir = gr.Textbox(value=str(Path.cwd() / "out"), label="Output dir")
                edges = gr.Radio(
                    ["feather", "matting"], value="feather", label="Edge quality",
                    info="feather = fast. matting = pymatting closed-form; best on hair/fine edges.",
                )
                with gr.Row():
                    feather = gr.Slider(0, 10, value=2, step=1, label="Feather (px)")
                    matting_band = gr.Slider(2, 24, value=8, step=1, label="Matting band (px)")
                with gr.Row():
                    include_bg = gr.Checkbox(value=True, label="Write bg.png")
                    include_css = gr.Checkbox(value=True, label="Write snippet.html")
                save_btn = gr.Button("Save layers", variant="primary")
                export_btn = gr.Button("Show points.json")
                spec_out = gr.Code(label="points.json", language="json")
                css_out = gr.Code(label="CSS snippet", language="html")

            with gr.Column(scale=3):
                preview = gr.Image(
                    label="Click to add points / draw box for the active layer",
                    type="numpy",
                    format="png",
                    interactive=False,
                    elem_classes=["big-preview"],
                )
                status = gr.Markdown("Upload an image to begin.")

        image_in.change(on_upload, [image_in, model, device, layers_state],
                        [preview, layers_state, active_layer, status, points_df])
        add_btn.click(on_add_layer, [new_name, layers_state],
                      [layers_state, active_layer, new_name, status])
        remove_btn.click(on_remove_layer, [active_layer, layers_state],
                        [layers_state, active_layer, preview, status, points_df])
        clear_btn.click(on_clear_points, [active_layer, layers_state],
                        [layers_state, preview, status, points_df])
        undo_btn.click(on_undo, [active_layer, layers_state],
                       [layers_state, preview, status, points_df])
        active_layer.change(on_active_layer_change, [active_layer, layers_state],
                            [preview, points_df])
        point_mode.change(on_mode_change, [point_mode, active_layer, layers_state], [preview])
        preview.select(on_image_click, [point_mode, active_layer, layers_state],
                       [layers_state, preview, status, points_df])
        points_df.input(on_points_df_change, [points_df, active_layer, layers_state],
                        [layers_state, preview, status])
        save_btn.click(on_save,
                       [out_dir, edges, feather, matting_band, include_bg, include_css, layers_state],
                       [status, css_out])
        export_btn.click(on_export_spec, [layers_state], [spec_out])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam2-hiera-large")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo = build_ui(args)
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
