#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_ner_predict.sh <mode>"
    exit
fi
mode=$1
MODEL_PATH='/home/cc/code/ner_poly_models'
out_dir=./output/$mode
python run_ner.py \
  --model_name_or_path $MODEL_PATH \
  --output_dir $out_dir \
  --test_file ../data/PolymerAbstracts/${mode}.json \
  --do_predict \
  --exp_name exp_$mode \
  --max_seq_len 512  \
  --metric_for_best_model f1
