# -*- coding: utf-8 -*-
"""
Minimal corr-grid visualizer for ESCNet.
- import と 1 行呼び出し用のユーティリティ。
- corr_grid (B,H_emb,W_emb,Ncls) を元画像サイズに拡大し、指定クラスだけをヒートマップで重畳保存。
- "関係ないクラスが赤くなる" 問題対策として、複数の正規化モードを用意。
"""
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# matplotlib は多くの環境に入っている前提。無ければ簡易カラーマップにフォールバック
try:
    import matplotlib.cm as cm
    _GET_CMAP = lambda name: cm.get_cmap(name)
except Exception:
    _GET_CMAP = lambda name: None

# ---------- 小道具 ----------
def _to_pil_from_tensor(img_t: torch.Tensor) -> Image.Image:
    """
    img_t: (3,H,W) float/uint8. 値域が[0,1]っぽければ255倍。
    """
    x = img_t.detach().float().cpu()
    if x.max() <= 1.5:
        x = x * 255.0
    x = x.clamp(0,255).byte().numpy()
    return Image.fromarray(np.moveaxis(x, 0, 2), mode="RGB")

def _load_label_map(label_map: Optional[Union[str, Dict, List]]):
    """
    dict({"11":"person"}), list(["bg","person",...]), または JSON/TXT のパス。
    TXT は "id: name" 形式に対応。
    """
    if label_map is None:
        return None, {}
    if isinstance(label_map, (dict, list)):
        obj = label_map
    else:
        p = str(label_map)
        with open(p, "r", encoding="utf-8") as f:
            first = f.read(1024); f.seek(0)
            if first.lstrip().startswith(("{","[")):
                obj = json.load(f)
            else:
                # id: name 形式
                mapping = {}
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("#"): continue
                    if ":" in line:
                        k,v = line.split(":",1)
                        mapping[str(int(k.strip()))] = v.strip()
                obj = mapping
    # name -> id & id -> name へ
    name2id, id2name = {}, {}
    if isinstance(obj, dict):
        for k,v in obj.items():
            id2name[int(k)] = str(v)
            name2id[str(v).lower()] = int(k)
    else:  # list
        for i,v in enumerate(obj):
            id2name[i] = str(v)
            name2id[str(v).lower()] = i
    return id2name, name2id

def _class_ids_from_any(classes: Optional[Sequence[Union[int,str]]],
                        name2id: Dict[str,int]) -> Optional[List[int]]:
    if classes is None:
        return None
    ids = []
    for c in classes:
        if isinstance(c, int):
            ids.append(c)
        else:
            cid = name2id.get(str(c).lower(), None)
            if cid is not None:
                ids.append(cid)
    return ids

def _normalize_map(m: torch.Tensor, mode: str = "zsig",
                   tau: float = 2.0, percentile: float = 95.0) -> torch.Tensor:
    """
    m: (H,W) float
    - "minmax": (m - min)/(max-min)
    - "zsig"  : z-score -> sigmoid(z/tau)   # 関係ないクラスは 0.5 付近に収束
    - "contrast": (m - mean).relu() / max   # 背景を抑えて局所ピークを強調
    - "percentile": m - P% を ReLU、その後 max で割る（ノイズ抑制に強い）
    返り値は [0,1]
    """
    m = m.float()
    if mode == "minmax":
        mn, mx = m.min(), m.max()
        return (m - mn) / (mx - mn + 1e-6)
    elif mode == "zsig":
        mu, sd = m.mean(), m.std()
        z = (m - mu) / (sd + 1e-6)
        s = torch.sigmoid(z / max(tau, 1e-6))
        return (s - s.min()) / (s.max() - s.min() + 1e-6)
    elif mode == "contrast":
        z = (m - m.mean()).clamp_min(0)
        return z / (z.max() + 1e-6)
    elif mode == "percentile":
        p = torch.quantile(m.reshape(-1), q=min(max(percentile/100.0,0.0),1.0))
        z = (m - p).clamp_min(0)
        return z / (z.max() + 1e-6)
    else:
        return _normalize_map(m, "minmax")

def _apply_colormap(hm: np.ndarray, cmap_name="jet") -> Image.Image:
    """
    hm: [H,W] (0..1)
    """
    H,W = hm.shape
    cmap = _GET_CMAP(cmap_name)
    if cmap is None:
        # 簡易 fall-back: 3ch グラデーション
        rgb = np.stack([hm, hm**0.5, 1.0-hm], axis=2)
        rgb = (rgb*255).clip(0,255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    rgba = cmap(hm)  # (H,W,4) float0..1
    rgb = (rgba[...,:3]*255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

# ---------- 外部呼び出し 1 行用 API ----------
@torch.no_grad()
def corrviz_from_model(
    corr_grid: torch.Tensor,
    batched_inputs: Sequence[dict],
    out_dir: str,
    *,
    classes: Optional[Sequence[Union[int,str]]] = None,
    label_map: Optional[Union[str,Dict,List]] = None,
    norm: str = "zsig",
    tau: float = 2.0,
    percentile: float = 95.0,
    alpha: float = 0.45,
    cmap: str = "jet",
    resize_mode: str = "bilinear",
    print_once: bool = True,
    # ---- ここから追加引数 ----
    lang_out: Optional[torch.Tensor] = None,
    channels: Optional[Union[str,Sequence[int]]] = None,  # "all" or [idx,...]
):
    """
    ESCNet.forward の中から 1 行で呼ぶための関数。
    corr_grid が与えられれば従来通り corr を、lang_out が与えられれば lang_out を可視化する。
    lang_out 形状: (B*Ncls, HW, C) / (B, Ncls, HW, C) / (B, Ncls, H_emb, W_emb, C) を想定。
    """
    # DDP rank0 以外はスキップ
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    id2name, name2id = _load_label_map(label_map)
    target_ids = _class_ids_from_any(classes, name2id) if classes is not None else None

    # ---------- 可視化対象の決定: lang_out 優先 ----------
    use_lang = (lang_out is not None)

    # 画像枚数
    B = len(batched_inputs)

    # 共通: 入力画像の取得
    def _get_img_pil(inp):
        fpath = inp.get("file_name", None)
        if fpath and os.path.exists(fpath):
            try:
                return Image.open(fpath).convert("RGB"), fpath
            except Exception:
                pass
        img_t = inp.get("image", None)
        if isinstance(img_t, torch.Tensor):
            x = img_t.detach().float().cpu()
            if x.max() <= 1.5:
                x = x * 255.0
            x = x.clamp(0,255).byte().numpy()
            pil = Image.fromarray(np.moveaxis(x,0,2), mode="RGB")
            return pil, fpath or ""
        # fallback
        H0 = int(inp.get("height", 480)); W0 = int(inp.get("width", 640))
        return Image.new("RGB", (W0,H0), (0,0,0)), fpath or ""

    # ---------- lang_out 可視化モード ----------
    if use_lang:
        t = lang_out
        # 形状を (B, Ncls, H_emb, W_emb, C) へ正規化
        if t.dim() == 3:                # (B*Ncls, HW, C)
            BN, HW, C = t.shape
            Ncls = max(BN // B, 1)
            g = int(round(HW ** 0.5))
            H_emb = W_emb = g
            t = t.reshape(B, Ncls, HW, C).reshape(B, Ncls, H_emb, W_emb, C)
        elif t.dim() == 4:              # (B, Ncls, HW, C)
            B2, Ncls, HW, C = t.shape
            g = int(round(HW ** 0.5))
            H_emb = W_emb = g
            t = t.reshape(B2, Ncls, H_emb, W_emb, C)
        elif t.dim() == 5:              # (B, Ncls, H_emb, W_emb, C)
            B2, Ncls, H_emb, W_emb, C = t.shape
            assert B2 == B, "lang_outのBがbatched_inputsと一致しません"
        else:
            raise ValueError(f"Unsupported lang_out shape: {tuple(t.shape)}")

        # チャンネル選択
        if isinstance(channels, str) and channels.lower() == "all":
            ch_list = list(range(int(t.shape[-1])))
        elif channels is None:
            ch_list = [0]  # 既定: とりあえず ch0 のみ
        else:
            ch_list = [int(c) for c in channels]

        # CPUへ（メモリ節約のため随時搬送）
        t = t.detach().float().cpu()  # (B,Ncls,H_emb,W_emb,C)

        for b in range(B):
            inp = batched_inputs[b]
            img_pil, fpath = _get_img_pil(inp)
            H0, W0 = img_pil.height, img_pil.width
            base = Path(fpath).name if fpath else f"batch{b:02d}.jpg"

            # (Ncls,H_emb,W_emb,C) -> (C,Ncls,H0,W0) に拡大しても良いが、
            # メモリ節約のため class/ch ごとにアップサンプリング
            cls_ids = target_ids if target_ids is not None else list(range(Ncls))
            for cid in cls_ids:
                if cid < 0 or cid >= Ncls: continue
                for ch in ch_list:
                    if ch < 0 or ch >= t.shape[-1]: continue
                    m = t[b, cid, :, :, ch].unsqueeze(0).unsqueeze(0)      # (1,1,H_emb,W_emb)
                    m_up = F.interpolate(m, size=(H0, W0), mode=resize_mode, align_corners=False)[0,0]

                    # 正規化
                    if norm == "zsig":
                        mu, sd = m_up.mean(), m_up.std()
                        z = (m_up - mu) / (sd + 1e-6)
                        s = torch.sigmoid(z / max(tau,1e-6))
                        mmn = (s - s.min()) / (s.max() - s.min() + 1e-6)
                    elif norm == "percentile":
                        p = torch.quantile(m_up.reshape(-1), q=min(max(percentile/100.0,0.0),1.0))
                        z = (m_up - p).clamp_min(0)
                        mmn = z / (z.max() + 1e-6)
                    elif norm == "contrast":
                        z = (m_up - m_up.mean()).clamp_min(0)
                        mmn = z / (z.max() + 1e-6)
                    else:  # "minmax"
                        mn, mx = m_up.min(), m_up.max()
                        mmn = (m_up - mn) / (mx - mn + 1e-6)

                    # 合成
                    hm = _apply_colormap(mmn.numpy(), cmap_name=cmap)
                    overlay = Image.blend(img_pil, hm, alpha=float(alpha))

                    # 小ラベル
                    try:
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(overlay)
                        cname = id2name.get(int(cid), str(cid)) if id2name else str(cid)
                        txt = f"{cid}: {cname}  | ch={ch}"
                        tw, th = draw.multiline_textbbox((0,0), txt)[2:]
                        draw.rectangle([2,2, 2+tw+6, 2+th+4], fill=(0,0,0,160))
                        draw.text((5,4), txt, fill=(255,255,255))
                    except Exception:
                        pass

                    out_name = f"{Path(base).stem}__{cid}_{(id2name.get(int(cid), '') if id2name else '').replace(' ','_')}__ch{ch:03d}.jpg"
                    overlay.save(os.path.join(out_dir, out_name))

        if print_once:
            print(f"[corrviz] saved LANG heatmaps to: {out_dir}  (norm={norm}, alpha={alpha})", flush=True)
        return

    # ---------- ここから従来の corr_grid 可視化（既存コードをそのまま） ----------
    B,H_emb,W_emb,Ncls = corr_grid.shape
    corr = corr_grid.detach().float().cpu()  # (B,H_emb,W_emb,Ncls)

    for b in range(B):
        inp = batched_inputs[b]
        # 画像取得
        img_pil, fpath = _get_img_pil(inp)
        H0, W0 = img_pil.height, img_pil.width

        # (H_emb,W_emb,Ncls) -> (1,Ncls,H_emb,W_emb) -> upsample -> (Ncls,H0,W0)
        m = corr[b].permute(2,0,1).unsqueeze(0)
        m_up = F.interpolate(m, size=(H0,W0), mode=resize_mode, align_corners=False).squeeze(0)

        cls_ids = target_ids if target_ids is not None else list(range(Ncls))
        base = Path(fpath).name if fpath else f"batch{b:02d}.jpg"

        for cid in cls_ids:
            if cid < 0 or cid >= Ncls: continue
            mm = m_up[cid]

            if norm == "zsig":
                mu, sd = mm.mean(), mm.std()
                z = (mm - mu) / (sd + 1e-6)
                s = torch.sigmoid(z / max(tau,1e-6))
                mmn = (s - s.min()) / (s.max() - s.min() + 1e-6)
            elif norm == "percentile":
                p = torch.quantile(mm.reshape(-1), q=min(max(percentile/100.0,0.0),1.0))
                z = (mm - p).clamp_min(0)
                mmn = z / (z.max() + 1e-6)
            elif norm == "contrast":
                z = (mm - mm.mean()).clamp_min(0)
                mmn = z / (z.max() + 1e-6)
            else:  # "minmax"
                mn, mx = mm.min(), mm.max()
                mmn = (mm - mn) / (mx - mn + 1e-6)

            hm = _apply_colormap(mmn.numpy(), cmap_name=cmap)
            overlay = Image.blend(img_pil, hm, alpha=float(alpha))

            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(overlay)
                name = id2name.get(int(cid), str(cid)) if id2name else str(cid)
                txt = f"{cid}: {name}"
                tw, th = draw.multiline_textbbox((0,0), txt)[2:]
                draw.rectangle([2,2, 2+tw+6, 2+th+4], fill=(0,0,0,160))
                draw.text((5,4), txt, fill=(255,255,255))
            except Exception:
                pass

            out_name = f"{Path(base).stem}__{cid}_{(id2name.get(int(cid), '') if id2name else '').replace(' ','_')}.jpg"
            overlay.save(os.path.join(out_dir, out_name))

    if print_once:
        print(f"[corrviz] saved heatmaps to: {out_dir}  (norm={norm}, alpha={alpha})", flush=True)
