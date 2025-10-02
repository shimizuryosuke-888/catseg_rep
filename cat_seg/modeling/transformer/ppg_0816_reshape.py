# 6.4 ppg_0816_reshape内部のsklearn等importを削除
# 修正後
# ppg.py
import torch
import torch.nn.functional as F
# from sklearn.cluster import KMeans
from typing import List, Tuple, Optional
# from ..utils.debug import dbg
import time





# ppg.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

"""
# ---- 追加: fast-pytorch-kmeans を優先 --------------------------------
try:
    from fast_pytorch_kmeans import KMeans as FPKMeans
    _FPK_OK = True
except Exception:
    _FPK_OK = False

print("FPK ", _FPK_OK)
# フォールバック用（任意）
try:
    from sklearn.cluster import KMeans
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False
print("Sk", _SKLEARN_OK)
# ----------------------------------------------------------------------
"""
# ----------------


@torch.no_grad()
def _grid_to_image_coords(
    coords_ij: torch.Tensor,          # (..., 2) [x, y]
    grid_size: Tuple[int, int],       # (H, W)
    image_size: Tuple[int, int],      # (H_img, W_img)
) -> torch.Tensor:
    H, W = grid_size
    Himg, Wimg = image_size
    x = coords_ij[..., 0].to(torch.float32) * (Wimg / W)
    y = coords_ij[..., 1].to(torch.float32) * (Himg / H)
    out = torch.stack([x.round().long(), y.round().long()], dim=-1)
    return out


@torch.no_grad()
def LinearSamplingPPG_regions(
    corr: torch.Tensor,          # (B, H, W, C_cls)
    k_pts: int = 5,
    thr: float = 0.5,
    image_size: Optional[Tuple[int,int]] = None,  # (H_img, W_img) for SAM
    use_kmeans: bool = True,       # 論文通り: True（クラスタ=KMeans）
    fast_no_kmeans_splits: int = 4 # フォールバック時の格子等分
):
    """
    返り値:
        coords        : (B, C, k_pts, 2)  [x, y] 画像ピクセル座標 (image_size 指定時)
        labels        : (B, C, k_pts)     すべて正例=1（パディングも1で統一）
        masks_regions : (B, C, k_pts, H, W)  各“領域”の2値マスク（特徴グリッド座標）
        bin_masks     : (B, C, H, W)      thr で二値化したクラス単位のマスク（参考用）
    """

    # time
    t_ppg_t0 = time.perf_counter()
    device = corr.device
    B, H, W, C = corr.shape

    # 1) 空間 softmax → 二値化
    corr_prob = F.softmax(corr.reshape(B, H*W, C), dim=1).reshape(B, H, W, C)
    bin_masks = (corr_prob >= thr).permute(0, 3, 1, 2).float()  # (B,C,H,W)

    coords_out = torch.zeros((B, C, k_pts, 2), dtype=torch.long,  device=device)
    labels_out = torch.ones((B, C, k_pts),     dtype=torch.long,  device=device)  # 正例=1
    masks_out  = torch.zeros((B, C, k_pts, H, W), dtype=torch.float32, device=device)

    # softmax_thresh
    t_ppg_thresh = time.perf_counter()

    for b in range(B):

        t_batch_start = time.perf_counter()

        prob_b = corr_prob[b]  # (H,W,C)
        for c_idx in range(C):
        
            t_cls_start = time.perf_counter()

            mask_map = bin_masks[b, c_idx]                 # (H,W)
            pts = mask_map.nonzero(as_tuple=False)         # (N_pos, 2) [y, x]
            if pts.numel() == 0:
                continue
            
            # --- 領域分割（KMeans 優先） ---
            if use_kmeans and pts.shape[0] > 1:
                n_clusters = min(k_pts, pts.shape[0])

                if _FPK_OK:
                    t_fpk_start = time.perf_counter()
                    # GPU上で完結
                    kmeans = FPKMeans(
                        n_clusters=n_clusters,
                        mode='euclidean',
                        max_iter=20,
                        tol=1e-4,
                        verbose=False,
                    )
                    # (N,) CUDA LongTensor
                    labels_cluster = kmeans.fit_predict(pts.float())
                    t_fpk_end = time.perf_counter()
                    # print(f"fpk: {t_fpk_end - t_fpk_start:.3f}s")
                elif _SKLEARN_OK:
                    # CPUフォールバック
                    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
                    labels_cluster = torch.from_numpy(
                        km.fit(pts.cpu().numpy()).labels_
                    ).to(device)
                else:
                    labels_cluster = None  # 下の等分割にフォールバック


                if labels_cluster is not None:
                    n_regions = n_clusters
                else:
                    # → 等分割へ
                    g = fast_no_kmeans_splits
                    while g*g > k_pts:
                        g -= 1
                    # y_id = (pts[:, 0] * g) // H
                    # x_id = (pts[:, 1] * g) // W
                    y_id = torch.div((pts[:, 0] * g), H, rounding_mode="floor").long()
                    x_id = torch.div(pts[:, 1] * g, W, rounding_mode="floor").long()
                    labels_cluster = (y_id * g + x_id)
                    n_regions = min(int(labels_cluster.max().item()) + 1, k_pts)
            else:
                # 等分割フォールバック
                g = fast_no_kmeans_splits
                while g*g > k_pts:
                    g -= 1
                # y_id = (pts[:, 0] * g) // H
                # x_id = (pts[:, 1] * g) // W
                y_id = torch.div((pts[:, 0] * g), H, rounding_mode="floor").long()
                x_id = torch.div(pts[:, 1] * g, W, rounding_mode="floor").long()
                labels_cluster = (y_id * g + x_id)
                n_regions = min(int(labels_cluster.max().item()) + 1, k_pts)
            t_cls_end = time.perf_counter()
            # print(f"cls: {t_cls_end -t_cls_start:.3f}s")
            # --- 代表点抽出（ベクトル化） ---
            # 全クラスタの領域マスクを一括で作る: (K,H,W)
            t_region_masks_start = time.perf_counter()
            region_masks = torch.zeros((n_regions, H, W), device=device, dtype=torch.float32)
            # 同じ (y,x) に複数点が来ても問題ないように accumulate=True
            ones = torch.ones(pts.size(0), device=device, dtype=torch.float32)
            region_masks = region_masks.index_put_(
                (labels_cluster.long(), pts[:, 0].long(), pts[:, 1].long()),
                ones,
                accumulate=True
            )
            region_masks = (region_masks > 0).float()

            # 空クラスタを安全に除去（理論上KMeansでは出にくいが念のため）
            valid = region_masks.reshape(n_regions, -1).any(dim=1)
            if not valid.all():
                region_masks = region_masks[valid]
                n_regions = region_masks.size(0)
                if n_regions == 0:
                    continue

            # 各領域ごとに “確率マップ×領域マスク” の argmax を取る（全領域をまとめて）
            pm = prob_b[:, :, c_idx]                    # (H,W)
            pm_flat = (region_masks * pm).reshape(n_regions, -1)  # (K, H*W)
            flat_idx = pm_flat.argmax(dim=1)                     # (K,)
            iy = torch.div(flat_idx, W, rounding_mode="floor")
            ix = torch.remainder(flat_idx, W)

            # k_pts に揃えて書き込み
            keep = min(k_pts, n_regions)
            coords_out[b, c_idx, :keep, 0] = ix[:keep]
            coords_out[b, c_idx, :keep, 1] = iy[:keep]
            masks_out[b, c_idx, :keep]    = region_masks[:keep]
            
            t_region_masks_end = time.perf_counter()
            # print(f"region masks: {t_region_masks_end - t_region_masks_start:.3f}s")
        t_batch_end = time.perf_counter()
        print(f"batch: {t_batch_end - t_batch_start:.3f}s")
    if image_size is not None:
        coords_out = _grid_to_image_coords(
            coords_out, grid_size=(H, W), image_size=image_size
        )

    # print(f"[TIMING] ppg_softmax: {t_ppg_t0 - t_ppg_thresh:.3f}s")

    return coords_out, labels_out, masks_out, bin_masks


# ppg_vectorized.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def LinearSamplingPPG_regions_vectorized(
    corr: torch.Tensor,                # (B, H, W, C)
    k_pts: int = 5,
    thr: float = 0.002,
    image_size: Optional[Tuple[int,int]] = (336, 336),  # (H_img, W_img)
    kmeans_iters: int = 7,
) :
    """
    論文の想定どおり:
      - クラスごと (各 B×C) に「空間 softmax 後のしきい値マスク」で有効画素を抽出
      - その有効画素の (x,y) を KMeans で k_pts クラスタに分割
      - 各クラスタから "確率最大" の代表点を選ぶ
      - 代表点座標 (画像座標系にスケール可)、クラスタ2値マスク、クラス2値マスクを返す
    すべて (B, C) をまとめてベクトル化。B や C に依存する for ループなし。
    """


    device = corr.device
    dtype = corr.dtype
    B, H, W, C = corr.shape
    P = H * W
    K = k_pts
    # ==== 修正1: FP16/FP32 で安全な「大きい数」を動的に決定（1e30 を使用しない）====
    BIG = torch.finfo(dtype).max / 4  # 例: fp16 ≈ 1.6e4, fp32 ≈ 8e37（どちらも有限）

    # ---- 1) 空間 softmax → 二値化（各クラスで空間方向に正規化） ----
    corr_prob = F.softmax(corr.reshape(B, P, C), dim=1).reshape(B, H, W, C)       # (B,H,W,C)
    prob_flat = corr_prob.permute(0, 3, 1, 2).reshape(B, C, P)              # (B,C,P)
    valid = (prob_flat >= thr)                                              # (B,C,P) bool

    # ---- 2) グリッド座標（[x,y]） ----
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=-1).reshape(P, 2).to(dtype)          # (P,2), [x,y] float

    # ---- 3) KMeans 初期化（各 B×C ごとに top-K 画素を代表初期値に） ----
    # 有効でない画素の確率を -inf にして topk から除外
    prob_masked = prob_flat.masked_fill(~valid, float("-inf"))              # (B,C,P)
    topk = min(K, P)
    _, idx0 = prob_masked.topk(topk, dim=2)                                 # (B,C,topk)
    centroids = coords[idx0]                                                # (B,C,topk,2)
    if topk < K:  # K > P の場合のパディング（最後の点を複製）
        pad = centroids[..., -1:, :].expand(B, C, K - topk, 2)
        centroids = torch.cat([centroids, pad], dim=2)                      # (B,C,K,2)

    # ---- 4) KMeans 反復（各 B×C を一括で）----
    # 反復の中でのみ固定回数 for（B や C 方向のループは一切なし）
    for _ in range(kmeans_iters):
        # 距離 (B,C,P,K)
        # diffs = X - mu
        diffs = coords[None, None, :, None, :] - centroids[:, :, None, :, :]  # (B,C,P,K,2)
        d2 = (diffs ** 2).sum(-1)                                             # (B,C,P,K)

        # 非有効画素は割り当て不能にする（大きな距離を加算）
        # 1e30はNaNを誘発する恐れあり
        # d2 = d2.masked_fill(~valid[:, :, :, None], 1e30)
        d2 = d2.masked_fill(~valid[:, :, :, None], BIG)

        # 割り当て（各画素 → 最近傍クラスタ）
        assign = d2.argmin(dim=3)                                             # (B,C,P)
        onehot = F.one_hot(assign, num_classes=K).to(dtype)                   # (B,C,P,K)
        onehot = onehot * valid[..., None].to(dtype)                          # 無効画素は0

        # 新しい重心（無効クラスタは元の重心を維持）
        denom = onehot.sum(dim=2).unsqueeze(-1)                                # (B,C,K,1)
        """
        修正前
            "bcpk,pc->bck2"の2が使用不可
            matmulの方が早い可能性があるため，matmulを使用
        # einsum: Σ_p onehot[b,c,p,k] * coords[p,:]
        num = torch.einsum("bcpk,pc->bck2", onehot, coords)                   # (B,C,K,2)
        """
        w = onehot.permute(0, 1, 3, 2).reshape(-1, P)                         # (B*C*K, P)
        num = (w @ coords.to(dtype)).reshape(B, C, K, 2)                         # (B,C,K,2)
        new_centroids = num / denom.clamp_min(1.0)
        keep_old = (denom.squeeze(-1) == 0)                                   # (B,C,K)
        centroids = torch.where(keep_old[..., None], centroids, new_centroids)

    # ---- 5) 最終割り当て → 領域マスク（各クラスタ） ----
    diffs = coords[None, None, :, None, :] - centroids[:, :, None, :, :]
    # 1e30はNaNを誘発する恐れあり
    # d2 = (diffs ** 2).sum(-1).masked_fill(~valid[:, :, :, None], 1e30)       # (B,C,P,K)
    d2 = (diffs ** 2).sum(-1).masked_fill(~valid[:, :, :, None], BIG)        # (B,C,P,K)
    assign = d2.argmin(dim=3)                                                # (B,C,P)
    onehot = F.one_hot(assign, num_classes=K).to(dtype)                       # (B,C,P,K)
    onehot = onehot * valid[..., None].to(dtype)

    # (B,C,K,P) -> (B,C,K,H,W)
    masks_regions = onehot.permute(0, 1, 3, 2).reshape(B, C, K, H, W)        # float

    # ---- 6) 各クラスタの代表点（確率最大の画素） ----
    masked_prob = (prob_flat[..., None] * onehot)                             # (B,C,P,K)
    flat_idx = masked_prob.argmax(dim=2)                                      # (B,C,K) in [0, P)
    # iy = flat_idx // W
    iy = torch.div(flat_idx, W, rounding_mode="floor")
    # ix = flat_idx %  W
    ix = torch.remainder(flat_idx, W)
    coords_out = torch.stack([ix, iy], dim=-1).long()                         # (B,C,K,2), [x,y]

    # SAM の正例ラベル（従来同様 1 を付与）
    labels = torch.ones((B, C, K), dtype=torch.long, device=device)          # (B,C,K)

    # クラス二値マスク（参考）
    bin_masks = valid.reshape(B, C, H, W).to(dtype)

    # 画像座標にスケール
    if image_size is not None:
        Himg, Wimg = image_size
        x = (coords_out[..., 0].to(torch.float32) * (Wimg / W)).round().long()
        y = (coords_out[..., 1].to(torch.float32) * (Himg / H)).round().long()
        coords_out = torch.stack([x, y], dim=-1)

    return coords_out, labels, masks_regions, bin_masks
