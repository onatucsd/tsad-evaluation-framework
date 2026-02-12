#export CUDA_VISIBLE_DEVICES=0
#echo "# path to me --------------->  ${0}"
python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model LightTS \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 55 \
  --c_out 55 \
  --top_k 3 \
  --batch_size 32 \
  --train_epochs 30 \
  --sc_function MoC \
  --th_idp 1 \
  --th_function Best-F \
  --ratio 100 \
  --baseline 0 