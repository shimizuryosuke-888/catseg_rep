# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_cat_seg_config

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .cat_seg_model import CATSeg
from .test_time_augmentation import SemanticSegmentorWithTTA

# 4.2 initファイルに追加
from .cat_seg_model_0909 import CATSeg_0909
from .cat_seg_model_0910 import CATSeg_0910
from .cat_seg_model_0911_Wrapper_review import CATSeg_0911_Wrapper_review
from .cat_seg_model_0917_Transformer_review import CATSeg_0917_Transformer_review