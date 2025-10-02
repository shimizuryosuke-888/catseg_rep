# cat_seg/utils/prompt_viz.py
import os, json
from typing import Sequence, Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# -------------------- small helpers --------------------
def _is_rank0() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_u8_rgb(img_t: torch.Tensor) -> np.ndarray:
    """[3,H,W] float(0..255想定) -> [H,W,3] u8"""
    x = img_t.detach().float().clamp(0, 255).cpu().numpy()
    return np.transpose(x, (1, 2, 0)).astype(np.uint8)

# 追加: 頭に置くヘルパ（_to_u8_rgb のすぐ下に）
def _robust_img_u8(img_t: torch.Tensor, bi: dict) -> np.ndarray:
    """
    images[b] が 0..1 や 平均減算後 になっていても可視化できるようにする:
      1) 0..1 のときは 255 倍
      2) それでも暗すぎる/ダイナミックレンジが小さいときは file_name から読み直し
    """
    x = img_t.detach().float().cpu()
    vmin, vmax = float(x.min()), float(x.max())
    # case-1: 0..1（少し余裕を見て判定）
    if 0.0 <= vmin and vmax <= 1.5:
        u8 = (x.clamp(0, 1) * 255.0).numpy()
        u8 = np.transpose(u8, (1, 2, 0)).astype(np.uint8)
        print("case1")
        return u8

    # 通常の 0..255 仮定
    u8 = _to_u8_rgb(img_t)

    # case-2: それでも暗すぎる（正規化済み画像など）
    if u8.max() < 30:  # だいたい真っ黒
        path = bi.get("file_name", None)
        if isinstance(path, str) and os.path.exists(path):
            try:
                print("case_try")
                return np.array(Image.open(path).convert("RGB"))
            except Exception:
                pass
        print("case2")
    return u8


def _load_label_map(path_or_obj: Union[str, dict, list, None]) -> dict:
    """
    受け付ける形式:
      - dict: {"11": "person", ...}
      - list: ["background", "wall", ...]  (indexがID)
      - str : 上のどちらかをJSONで保存したファイルパス
    """
    if path_or_obj is None:
        return {}
    if isinstance(path_or_obj, dict):
        return {str(k): str(v) for k, v in path_or_obj.items()}
    if isinstance(path_or_obj, list):
        return {str(i): str(n) for i, n in enumerate(path_or_obj)}
    if isinstance(path_or_obj, str) and os.path.exists(path_or_obj):
        with open(path_or_obj, "r") as f:
            obj = json.load(f)
        return _load_label_map(obj)
    return {}

def _name_to_ids(names_csv: str, label_map: dict) -> List[int]:
    """'person,car' → [ID,...]（無い名前は無視）"""
    inv = {v: int(k) for k, v in label_map.items()}
    ids = []
    for nm in [s.strip() for s in (names_csv or "").split(",") if s.strip()]:
        if nm in inv:
            ids.append(inv[nm])
    return ids

def _parse_ids(ids_csv: str) -> List[int]:
    return [int(s) for s in (ids_csv or "").split(",") if s.strip().isdigit()]

def _draw_points(draw: ImageDraw.ImageDraw, pts_xy: np.ndarray, color=(255, 255, 0), r=4):
    """pts_xy: [K,2] (x,y)"""
    for (x, y) in pts_xy:
        x = float(x); y = float(y)
        print("debug: prompt_viz.py _draw_points")
        print("x = ", x)
        print("y = ", y)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2, fill=None)

def _overlay_mask_u8(base_u8: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), alpha=0.35) -> np.ndarray:
    """mask01: [H,W] float in [0,1]"""
    h, w = mask01.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:] = np.array(color, dtype=np.uint8)
    m = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
    m3 = np.stack([m, m, m], axis=-1)
    # alpha-blend only where mask>0
    return np.where(m3 > 0, (1 - alpha) * base_u8 + alpha * overlay, base_u8).astype(np.uint8)

# -------------------- main API --------------------
@torch.no_grad()
def dump_prompts_once(
    images: Sequence[torch.Tensor],                 # list of [3,H,W] float(0..255)
    batched_inputs: Sequence[dict],                 # Detectron2 の入力（file_name を使う）
    coords_img: torch.Tensor,                       # [B, C, K, 2] (画像座標の点プロンプト)
    masks_regions: torch.Tensor,                    # [B, C, K, Hg, Wg] (特徴グリッド上の“マスクプロンプト群”)
    *,
    grid_size: Optional[Tuple[int, int]] = None,    # (Hg,Wg) を手動指定したい時（通常は不要）
    display_hw: Optional[Tuple[int, int]] = None,  # ← 追加: 表示用 (H, W)
    outdir: str = "./prompt_viz",
    alpha: float = 0.35,
    mask_color: Tuple[int,int,int] = (0, 255, 0),
    point_color: Tuple[int,int,int] = (255, 255, 0),
    point_radius: int = 4,
    class_ids: Optional[Sequence[int]] = None,      # 例: [11,94]
    class_names: Optional[str] = None,              # 例: "person,car"（label_map必須）
    label_map: Union[str, dict, list, None] = None, # {"11":"person",...} or path
    save_all_if_no_target: bool = False,            # 指定が無いとき全クラスを出すか（既定False）
) -> None:
    """
    各 画像 × 指定クラス につき 1枚: <元名>__<cid>_<name>.jpg を outdir に保存します。
    * 点は全K個を描画
    * マスクは各K領域の“和”を作って薄く着色（= SAMの mask prompt のイメージ）

    期待shapeは、PPG系実装の戻り値と一致しています：
      coords_img: (B,C,K,2), masks_regions: (B,C,K,Hg,Wg)  :contentReference[oaicite:0]{index=0}
    """
    if not _is_rank0():
        return

    B = len(images)
    assert isinstance(coords_img, torch.Tensor) and isinstance(masks_regions, torch.Tensor)
    assert coords_img.dim() == 4 and masks_regions.dim() == 5, "coords_img:[B,C,K,2], masks_regions:[B,C,K,Hg,Wg] を渡してください"

    # ラベル名解決
    lm = _load_label_map(label_map)
    want_ids = list(class_ids or [])
    if class_names:
        want_ids += _name_to_ids(class_names, lm)
    # 重複除去
    want_ids = sorted(set(int(x) for x in want_ids))

    _ensure_dir(outdir)

    # バッチごと
    for b in range(B):
        # ① interpolate 用に float32 へ明示的に変換（Byte対策）
        img_t = images[b].to(torch.float32)  # [3,H,W]

        # ② 表示用にリサイズ（指定があるとき）
        if display_hw is not None:
            Ht, Wt = int(display_hw[0]), int(display_hw[1])
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=(Ht, Wt),
                mode="bilinear", align_corners=False
            )[0]

        # ③ 可視化の元画像は「リサイズ済みの img_t」を使う
        img_u8 = _robust_img_u8(img_t, batched_inputs[b])

        H, W = img_u8.shape[:2]
        pil = Image.fromarray(img_u8.copy())
        drw = ImageDraw.Draw(pil)

        # ★ここから2行追加（画像サイズと元ファイル名を表示）
        file_base = os.path.basename(batched_inputs[b].get("file_name", f"img{b}.jpg"))
        print(f"[prompt_viz] image[{b}] file={file_base} size={W}x{H} (W×H)")

        file_base = os.path.basename(batched_inputs[b].get("file_name", f"img{b}.jpg"))
        stem, _ = os.path.splitext(file_base)

        _, C, K, _ = coords_img.shape
        Hg, Wg = masks_regions.shape[-2:]
        if grid_size is not None:
            Hg, Wg = grid_size
        
        # ★ここに1行追加（グリッド→画像サイズの対応）
        print(f"[prompt_viz] mask_prompt_grid {Hg}x{Wg} -> upsample to {W}x{H}")

        # 対象クラス集合
        if want_ids:
            targets = [cid for cid in want_ids if 0 <= cid < C]
        else:
            if not save_all_if_no_target:
                # 何も指定が無ければスキップ
                continue
            targets = list(range(C))

        # マスクを画像サイズへ
        # classごとの領域“和”を作ってからアップサンプル → (B=1で処理)
        for cid in targets:
            name = lm.get(str(cid), str(cid))
            print("debug: prompt_viz.py class_name ", name)

            # ----- 1) マスクプロンプト（K領域の和） -----
            m = masks_regions[b, cid].any(dim=0).float().unsqueeze(0).unsqueeze(0)  # [1,1,Hg,Wg]
            m_up = F.interpolate(m, size=(H, W), mode="nearest")[0, 0].cpu().numpy()  # 0/1

            # 背景に薄く塗る
            blended = _overlay_mask_u8(np.array(pil), m_up, color=mask_color, alpha=alpha)
            pil_masked = Image.fromarray(blended)
            draw2 = ImageDraw.Draw(pil_masked)

            # ----- 2) 点プロンプト（K点） -----
            pts = coords_img[b, cid].cpu().numpy()  # [K,2] (x,y)
            _draw_points(draw2, pts, color=point_color, r=point_radius)

            # 左上にクラス名
            txt = f"{cid}: {name}"
            tw = draw2.textlength(txt)
            pad = 6; h = 14
            draw2.rectangle([5, 5, 5 + tw + pad*2, 5 + h + pad], fill=(0,0,0,160))
            draw2.text((5+pad, 5+pad//2), txt, fill=(255,255,255))

            out_path = os.path.join(outdir, f"{stem}__{cid}_{name}.jpg")
            pil_masked.save(out_path)
            print(f"[prompt_viz] saved: {out_path}")
