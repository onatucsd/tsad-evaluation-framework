#export CUDA_VISIBLE_DEVICES=1
#echo "# path to me --------------->  ${0}"
python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/WADI \
  --model_id WADI \
  --model MICN \
  --data WADI \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 123 \
  --c_out 123 \
  --batch_size 128 \
  --train_epochs 20 \
  --sc_function MoC \
  --th_idp 1 \
  --th_function Best-F \
  --ratio 100 \
  --baseline 0 