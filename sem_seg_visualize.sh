#!/bin/sh


output_dir=$1



if [ -z $output_dir ]
then
    echo "No output_dir! ex.output/0901"
    exit 0
fi



#ADE20k-150
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/eval/ade150/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/eval/ade150/visualized \
  --alpha 0.5 \
  --label-map datasets/ade150.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

#ADE20k-847
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/eval/ade847/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/eval/ade847/visualized \
  --alpha 0.5 \
  --label-map datasets/ade847.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

#Pascal VOC
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/eval/pascal_voc/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/eval/pascal_voc/visualized \
  --alpha 0.5 \
  --label-map datasets/voc20.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

#Pascal VOC-b
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/eval/pascal_voc_b/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/eval/pascal_voc_b/visualized \
  --alpha 0.5 \
  --label-map datasets/voc20b.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

#Pascal Context 59
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/eval/pascal_context_59/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/eval/pascal_context_59/visualized \
  --alpha 0.5 \
  --label-map datasets/pc59.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

#Pascal Context 459
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/eval/pascal_context_459/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/eval/pascal_context_459/visualized \
  --alpha 0.5 \
  --label-map datasets/pc459.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

#coco-stuff
python visualizer_sem_seg_predictions_v4.py \
  --json $output_dir/inference/sem_seg_predictions.json \
  --image-root /home/jaxa/shimizu/catseg_rep/CAT-Seg \
  --out $output_dir/visualized \
  --alpha 0.5 \
  --label-map datasets/coco_2.json \
  --max-images 10 \
  --topk-fill 10 \
  --min-area-ratio 0.003

