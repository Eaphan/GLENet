exp_id=exp20
tag=exp20_aug_shift01_fix_gpu2
epoch=400
gpu=0

save_path=output/${exp_id}/${tag}/eval/epoch_${epoch}/val/default/final_result/data

CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag}  --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val1.pkl
CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag}  --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val2.pkl
CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag}  --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val3.pkl
CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag}  --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val4.pkl
CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag}  --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val5.pkl

python evaluate.py $exp_id $tag $epoch
# python eval.py $exp_id $tag $epoch
