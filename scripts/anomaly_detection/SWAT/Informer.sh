#export CUDA_VISIBLE_DEVICES=0
echo "# path to me --------------->  ${0}"
python3 -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model Informer \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --batch_size 128 \
  --train_epochs 3 \
  --anomaly_ratio 10 \
  --sc_function MoC \
  --th_idp 0 \
  --th_function Best-F \
  --ratio 90 \
  --baseline 0 
