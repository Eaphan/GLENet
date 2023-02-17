exp_id=$1
tag=$2
epoch=$3
gpu=$4

save_path=output/${exp_id}/${tag}/eval/epoch_${epoch}/val/default/final_result/data

for iter in `seq 1 30`;do
    CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag}  --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth \
         --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val${iter}.pkl
done


