# utils/debug.py

"""
memo: import
    inspect:
        呼び出し元の行番号を取得
    textwrap:
        今は使っていないが，長いメッセージが出たときに，折り返しができるようになる
"""
import torch, inspect, textwrap, os
import numpy as np

def _stat_each_dim(t: torch.Tensor):
    parts = []
    for d in range(t.ndim):
        # 「軸 d 以外」を一気に縮約
        other_dims = tuple(i for i in range(t.ndim) if i != d)
        # まず他の軸を全部まとめて最小値テンソルを得る → shape=(size_of_axis_d,)
        red_min = t.amin(dim=other_dims)
        red_max = t.amax(dim=other_dims)
        # そのベクトルの中で最小・最大を取る
        mn = red_min.min().item()
        mx = red_max.max().item()
        parts.append(f"ax{d} {mn:.2f}/{mx:.2f}")
    return " | ".join(parts)



def _stat_reduce(t: torch.Tensor, dims):
    """dims 指定あり→その軸でまとめた min/max"""
    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(sorted(set(dims)))
    mn = t
    mx = t
    for d in dims:
        mn = mn.amin(dim=d, keepdim=True)
        mx = mx.amax(dim=d, keepdim=True)
    return f"{mn.mean():.2f}/{mx.mean():.2f} (dims={dims})"

def dbg(*, dims=None, **tensors):
    """
    >>> dbg(x=some_tensor, y=other)
    x: torch.Size([4, 128, 24, 24])  | dtype=float32 | device=cuda:0 | min=-1.3 max=2.1
    y: torch.Size([4, 1024])         | dtype=float16 | device=cuda:0 | min=0.0  max=1.0
    """

    """
    memo:
        def dbg(**tensors):
            キーワード引数で受け取る
        process:
            frame = inspect.currentframe().f_back
                dbgを呼んだ一つ前のフレームオブジェクト
                フレームオブジェクト...?
                    関数とかのオブジェクトかな？
            
            lineno = frame.f_lineno:
                frameオブジェクトのプログラム内行番号取得

            file:
                呼び出し元ファイル名
    """


    frame = inspect.currentframe().f_back
    lineno = frame.f_lineno
    file = os.path.basename(inspect.getfile(frame))

    
    dspec = dims 
    # 引数として受け取ったものを順番に取り出し，処理する．
    for name, t in tensors.items():
        # テンソルであった場合に以下の処理を行う．
        if isinstance(t, torch.Tensor):
            # 要素数numel()が0でないなら，最大値と最小値を2f 2桁まで表示
            # データがおかしな値になっていないかチェック．
            ## NaNとか0とか
            # stat = _stat_each_dim(t) if dspec is None else _stat_reduce(t, dspec)
            # 行番号，変数名，データ形状，データ型，デバイスをきれいに表示
            print(f"[L{lineno:4} {file}] {name:<15}: {tuple(t.shape)!s:<18}"
                  f"| {str(t.dtype).replace('torch.',''):<9}"
                  f"| {t.device} ")
            print(t)
        else:
            # テンソルでない場合，変数名と中身をそのまま表示
            # print(f"[L{lineno:4}] {name} {np.shape(t)}: {t}")
            try:
                l = len(t)
                info = f"(len={l})"
            except Exception:
                info = "(?)"
            print(F"[L{lineno:4}] {name:<15} {info}: {t}")
