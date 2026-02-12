#export CUDA_VISIBLE_DEVICES=0
#echo "# path to me --------------->  ${0}"
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/NIPS_TS_Swan \
  --model_id NIPS_TS_Swan \
  --model PatchTST \
  --data NIPS_TS_Swan \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 38 \
  --c_out 38 \
  --batch_size 128 \
  --train_epochs 20 \
  --sc_function MoC \
  --th_idp 1 \
  --th_function Best-F \
  --ratio 100 \
  --baseline 0 
