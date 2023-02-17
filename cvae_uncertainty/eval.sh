exp_id=exp20
tag=debug # fsr_e1600 fsr_64
epoch=400
gpu=3

save_path=output/${exp_id}/${tag}/eval/epoch_${epoch}/val/default/final_result/data

for iter in `seq 1 30`;do
     CUDA_VISIBLE_DEVICES=${gpu} python test.py --cfg_file cfgs/${exp_id}.yaml --extra_tag ${tag} \
          --ckpt output/${exp_id}/${tag}/ckpt/checkpoint_epoch_${epoch}.pth --workers 1 --save_to_file && cp ${save_path}/result.pkl ${save_path}/result_val${iter}.pkl
done

#python evaluate.py $exp_id $tag $epoch
python eval.py $exp_id $tag $epoch &> logs/${exp_id}_${tag}_eval.log

# 
# scale_flip_e1600 373.173
# fsr_128 761.061, 736.362
# fsr_64 217.990, 193.058
# fsr_van 576.624, 597.301
# fsr_lr0.001 1373.725
# redo 479.931, 496.107, 490.369, 358.832, 442.043

# fsr_64_redo
# fsr_32
# fsr_256
# eval fsr_van, with config van
