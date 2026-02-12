export CUDA_VISIBLE_DEVICES=1
#echo "# path to me --------------->  ${0}"
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model GPT4TS \
  --data SMAP \
   --features M \
  --seq_len 100 \
  --pred_len 0 \
    --gpt_layers 6 \
  --d_model 768 \
  --d_ff 8 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 25 \
  --c_out 25 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --sc_function MoC \
  --th_idp 1 \
  --th_function Best-F \
  --ratio 100 \
  --baseline 0 