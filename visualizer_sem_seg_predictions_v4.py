# visualizer_sem_seg_predictions_v4.py
import os, json, argparse, random
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pycocotools.mask as maskutil

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# 任意の形式のラベルマップを {int id: str name} に正規化
def normalize_label_map(raw):
    mapping = {}
    if raw is None:
        return mapping

    # {"categories":[{id,name,...}, ...]}
    if isinstance(raw, dict) and "categories" in raw and isinstance(raw["categories"], list):
        for it in raw["categories"]:
            cid = it.get("id", it.get("index"))
            name = it.get("name", it.get("label", it.get("synonyms", it.get("syn"))))
            if isinstance(name, list):
                name = name[0]
            if cid is not None and name is not None:
                mapping[int(cid)] = str(name)
        return mapping

    # {"0":"wall", ...} / {"0":{"name":"wall"}, ...}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                cid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                name = v.get("name", v.get("label"))
                if isinstance(name, list):
                    name = name[0]
            else:
                name = v
            if name is not None:
                mapping[cid] = str(name)
        return mapping

    # [{"id":0,"name":"wall"}, ...] / ["wall","building",...]
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            for i, it in enumerate(raw):
                cid = it.get("id", it.get("index", i))
                name = it.get("name", it.get("label", it.get("synonyms", it.get("syn"))))
                if isinstance(name, list):
                    name = name[0]
                mapping[int(cid)] = str(name)
        else:
            for i, name in enumerate(raw):
                mapping[i] = str(name)
        return mapping

    return mapping

def try_load_image(root, file_name):
    # 絶対パス優先。相対なら root と結合。root がプロジェクト直下のときの冗長パスにも対応
    if os.path.isabs(file_name):
        cand = [file_name]
    else:
        cand = [
            os.path.join(root, file_name),
            os.path.join(os.path.dirname(root.rstrip("/")), file_name),
            os.path.join(root, os.path.basename(file_name)),
        ]
    for p in cand:
        if os.path.exists(p):
            return Image.open(p).convert("RGB"), p
    print(f"[WARN] image not found for '{file_name}'")
    return None, (cand[0] if cand else file_name)

def decode_rle(rle):
    m = maskutil.decode(rle)  # (H,W) or (H,W,K)
    if m.ndim == 3:
        return [m[..., k].astype(np.uint8) for k in range(m.shape[2])]
    return [m.astype(np.uint8)]

def seeded_color(cid):
    rnd = random.Random(int(cid) * 9973 + 12345)
    r = rnd.randint(40, 230); g = rnd.randint(40, 230); b = rnd.randint(40, 230)
    return (r, g, b)

def overlay_mask(img_rgb, mask, color, alpha=0.45):
    """img_rgb: PIL RGB (or None), mask: np.uint8 [H,W]{0,1}"""
    h, w = mask.shape
    base = img_rgb.copy() if img_rgb is not None else Image.new("RGB", (w, h), (0, 0, 0))
    overlay = Image.new("RGB", base.size, color)
    # 透明度はマスクの白さに比例（0/1 * alpha）
    m = Image.fromarray((mask * int(alpha * 255)).astype(np.uint8), mode="L")
    return Image.composite(overlay, base, m)

def mask_boundary(mask, width=2):
    """0/1 mask -> boolean boundary image（4近傍・簡易）。"""
    m = mask.astype(bool)
    diff_v = np.zeros_like(m); diff_h = np.zeros_like(m)
    diff_v[1:, :] = m[1:, :] ^ m[:-1, :]
    diff_h[:, 1:] = m[:, 1:] ^ m[:, :-1]
    edge = diff_v | diff_h
    e = edge.copy()
    for _ in range(max(0, width-1)):
        e |= np.pad(edge[1:, :], ((0,1),(0,0))) | np.pad(edge[:-1, :], ((1,0),(0,0))) \
           | np.pad(edge[:, 1:], ((0,0),(0,1))) | np.pad(edge[:, :-1], ((0,0),(1,0)))
        edge = e
    return e

def draw_outline(img_rgb, mask, color, width=2):
    edge = mask_boundary(mask, width)
    edge_img = Image.fromarray((edge * 255).astype(np.uint8), mode="L")
    col = Image.new("RGB", img_rgb.size, color)
    return Image.composite(col, img_rgb, edge_img)

def draw_inline_label(canvas, mask, text, color, font_size=None):
    """最大連結成分の重心近辺に小さなラベルを描く（簡易）。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    x = int(xs.mean()); y = int(ys.mean())
    w, h = canvas.size
    x = min(max(10, x), w-10); y = min(max(10, y), h-10)

    draw = ImageDraw.Draw(canvas)
    fs = font_size or max(12, int(min(w, h) * 0.02))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fs)
    except Exception:
        font = ImageFont.load_default()

    tw = draw.textlength(text, font=font)
    th = fs + 2
    pad = 4
    x0 = max(0, min(w - (tw + 2*pad), x - (tw // 2)))
    y0 = max(0, y - th - 6)

    draw.rectangle([x0, y0, x0 + tw + 2*pad, y0 + th], fill=(0, 0, 0, 180))
    draw.rectangle([x0+2, y0+2, x0+fs-2, y0+th-2], fill=color)
    draw.text((x0 + fs + 2, y0 + 1), text, fill=(255, 255, 255), font=font)

def build_per_image(records):
    per_cls = []
    for r in records:
        cid = r["category_id"]
        for m in decode_rle(r["segmentation"]):
            area = int(m.sum())
            per_cls.append({"cid": cid, "mask": m.astype(np.uint8), "area": area})
    return per_cls

def compose_and_save(
    file_name, per_cls, img_root, out_dir, order, alpha, label_map,
    min_area_px, min_area_ratio, topk_fill, outline_width=2,
    inline_labels=True, font_size=None
):
    # 画像ロード
    img, resolved = try_load_image(img_root, file_name)
    if img is None:
        if per_cls:
            h, w = per_cls[0]["mask"].shape
        else:
            h, w = 400, 400
        img = Image.new("RGB", (w, h), (0, 0, 0))
    H, W = img.height, img.width

    # 最小エリア（比率→px）
    if min_area_px <= 0 and min_area_ratio > 0:
        min_area_px = int(H * W * min_area_ratio)

    # フィルタ＆並べ替え
    per_cls = [p for p in per_cls if p["area"] >= min_area_px]
    if order == "area_desc":
        per_cls.sort(key=lambda x: x["area"], reverse=True)
    else:
        per_cls.sort(key=lambda x: x["area"])

    # 上位のみ塗って、全領域に輪郭
    per_desc = sorted(per_cls, key=lambda x: x["area"], reverse=True)
    fill_set = set([id(x) for x in per_desc[:max(0, topk_fill)]])

    canvas = img.copy()
    for p in per_desc:  # 大きい順に重ねる
        cid, mask = p["cid"], p["mask"]
        color = seeded_color(cid)

        if id(p) in fill_set:
            canvas = overlay_mask(canvas, mask, color, alpha=alpha)
        canvas = draw_outline(canvas, mask, color, width=outline_width)

        if inline_labels and id(p) in fill_set:
            name = label_map.get(int(cid), str(cid)) if label_map else str(cid)
            draw_inline_label(canvas, mask.astype(bool), name, color, font_size=font_size)

    # 保存
    rel = file_name.replace("/", "_")
    out_path = os.path.join(out_dir, rel)
    ensure_dir(out_path.rsplit("/", 1)[0])
    canvas.save(out_path)
    print(f"[saved] {out_path}  (src={resolved})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--order", choices=["area_asc","area_desc"], default="area_desc")
    ap.add_argument("--alpha", type=float, default=0.40)
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--min-area-px", type=int, default=0)
    ap.add_argument("--min-area-ratio", type=float, default=0.0)
    ap.add_argument("--topk-fill", type=int, default=6)
    ap.add_argument("--outline-width", type=int, default=2)
    ap.add_argument("--font-size", type=int, default=0, help="0なら自動")
    args = ap.parse_args()

    preds = load_json(args.json)
    raw_lm = load_json(args.label_map) if args.label_map and os.path.exists(args.label_map) else None
    label_map = normalize_label_map(raw_lm)
    print(f"[label_map] entries = {len(label_map)}")

    by_img = defaultdict(list)
    for r in preds:
        by_img[r["file_name"]].append(r)

    count = 0
    for file_name, recs in by_img.items():
        per_cls = build_per_image(recs)
        if len(per_cls) == 0:
            continue
        compose_and_save(
            file_name, per_cls, args.image_root, args.out, args.order, args.alpha, label_map,
            min_area_px=args.min_area_px,
            min_area_ratio=args.min_area_ratio,
            topk_fill=args.topk_fill,
            outline_width=args.outline_width,
            inline_labels=True,
            font_size=(args.font_size if args.font_size > 0 else None),
        )
        count += 1
        if args.max_images and count >= args.max_images:
            break

if __name__ == "__main__":
    main()
