# cat_seg/esc_net/sam_block.py

# samをbuildし，必要な処理を取り出す．

import torch.nn as nn
import segment_anything

# checkpointsのパス指定
_SAM_CKPT = "/home/jaxa/shimizu/catseg_rep/CAT-Seg/pretrained/sam_vit_l_0b3195.pth"  

def build_sam_block(freeze: bool = False) -> nn.Module:
    """ImageEncoder を捨て，PromptEncoder+MaskDecoder だけ返す．"""
    # CLIP画像サイズ，埋め込みに適応させたbuild_samを自作
    # それを用いてsamをbuild
    # 元のbuild
    # vithではなく，vitlとしてCLIPにあわせた方がいい
    sam_full = segment_anything.build_sam_vit_l(checkpoint=_SAM_CKPT)

    # 不要部を削除してメモリ節約
    # SamPredictorを用いる場合は，image_encoderを残しておく．
    del sam_full.image_encoder
    
    # mask_decoder内の使用しない処理を削除
    del sam_full.mask_decoder.output_upscaling
    del sam_full.mask_decoder.output_hypernetworks_mlps
    del sam_full.mask_decoder.iou_prediction_head

    sam_block = nn.ModuleDict({
        "prompt_encoder": sam_full.prompt_encoder,
        "mask_decoder"  : sam_full.mask_decoder,
    })

    if freeze:
        sam_block.eval()
        for p in sam_block.parameters():
            p.requires_grad = False
        print("sa_block freeze ", freeze)
    else:
        print("sam_block freeze ", freeze)



    return sam_block

def build_sam_full_block(freeze: bool = True) -> nn.Module:
    """ImageEncoder を捨て，PromptEncoder+MaskDecoder だけ返す．"""
    # CLIP画像サイズ，埋め込みに適応させたbuild_samを自作
    # それを用いてsamをbuild
    # 元のbuild
    # vithではなく，vitlとしてCLIPにあわせた方がいい
    sam_full = segment_anything.build_sam_vit_l(checkpoint=_SAM_CKPT)

    # 不要部を削除してメモリ節約
    # SamPredictorを用いる場合は，image_encoderを残しておく．
    # del sam_full.image_encoder

    sam_block = nn.ModuleDict({
        "prompt_encoder": sam_full.prompt_encoder,
        "mask_decoder"  : sam_full.mask_decoder,
    })

    if freeze:
        sam_block.eval()
        for p in sam_block.parameters():
            p.requires_grad = False

    # debug modelの中身調査
    """
    for n, p in sam_block.named_parameters():
        print(n, p.shape)
    """

    return sam_full


def build_sam_block_CLIP_image_size(freeze: bool = True, CLIP_image_size: int = None, CLIP_vit_patch_size: int = None) -> nn.Module:
    """ImageEncoder を捨て，PromptEncoder+MaskDecoder だけ返す．"""
    # CLIP画像サイズ，埋め込みに適応させたbuild_samを自作
    # それを用いてsamをbuild
    sam_full = segment_anything.build_sam_vit_l_CLIP(checkpoint=_SAM_CKPT, CLIP_image_size=CLIP_image_size,CLIP_vit_patch_size=CLIP_vit_patch_size)
    # 元のbuild
    # sam_full = segment_anything.build_sam_vit_h(checkpoint=_SAM_CKPT)

    # 不要部を削除してメモリ節約
    del sam_full.image_encoder

    sam_block = nn.ModuleDict({
        "prompt_encoder": sam_full.prompt_encoder,
        "mask_decoder"  : sam_full.mask_decoder,
    })

    if freeze:
        sam_block.eval()
        for p in sam_block.parameters():
            p.requires_grad = False

    # debug modelの中身調査
    """
    for n, p in sam_block.named_parameters():
        print(n, p.shape)
    """

    return sam_block

if __name__ == "__main__":
    build_sam_block(freeze=False)