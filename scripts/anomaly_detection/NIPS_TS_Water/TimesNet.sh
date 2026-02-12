#export CUDA_VISIBLE_DEVICES=1
#echo "# path to me --------------->  ${0}"
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/NIPS_TS_Water \
  --model_id NIPS_TS_Water \
  --model TimesNet \
  --data NIPS_TS_Water \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --enc_in 9 \
  --c_out 9 \
  --top_k 3 \
  --batch_size 128 \
  --train_epochs 20 \
  --sc_function MoC \
  --th_idp 1 \
  --th_function Best-F \
  --ratio 100 \
  --baseline 0 