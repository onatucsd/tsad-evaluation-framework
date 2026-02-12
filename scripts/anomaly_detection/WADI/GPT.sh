#export CUDA_VISIBLE_DEVICES=2
#echo "# path to me --------------->  ${0}"
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/WADI \
  --model_id WADI \
  --model GPT4TS \
  --data WADI \
   --features M \
  --seq_len 100 \
  --pred_len 0 \
    --gpt_layers 6 \
  --d_model 768 \
  --d_ff 8 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 123 \
  --c_out 123 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --sc_function MoC \
  --th_idp 1 \
  --th_function Best-F \
  --ratio 100 \
  --baseline 0 