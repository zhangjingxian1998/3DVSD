# The name of experiment 通过预训练, 使模型对于 <OBJ> 和 <REL> 进行学习
name=VLT5_replace_gt_extra_id

output=VL-T5/snap/VSD_3D/final/vsd2/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    VL-T5/src/VSD_3D.py \
        --distributed --multiGPU --fp16 \
        --data VSDv2 \
        --train train \
        --valid val \
        --test test\
        --batch_size 16 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 16 \
        --max_text_length 40 \
        --clip_grad_norm 1.0 \
        --n_ground 4 \
        --backbone 'VL-T5/t5-base' \
        --output $output ${@:2} \
        --load 'VL-T5/snap/pretrain/VLT5/Epoch30' \
        --num_beams 5 \
        --valid_batch_size 100 \
        --save_result_path 'save_img_vsd2' \
        --use_prefix \
        # --VL_pretrain \
        # --load 'VL-T5/snap/VSD_3D/pretrain/VLT5/BEST' \
        # --get_rel \
        # --test_only \
        # --replace_rel