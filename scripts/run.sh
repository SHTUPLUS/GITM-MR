model_name="RCRN"
dat=`date "+%Y-%m-%d-%H-%M-%S"`
output_dir="output/eval_${model_name}_${dat}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_refdet.py --num-gpus 4 --config-file configs/RCRN_len11.yaml \
  --config configs/train-ng-base-1gpu.json --eval-only --resume OUTPUT_DIR $output_dir