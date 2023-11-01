export CUDA_VISIBLE_DEVICES=2

# python tools/train.py configs/recognition/aim/vitclip_base_hmdb51.py

# bash tools/dist_train.sh configs/recognition/aim/vitclip_base_sthv2.py

# python tools/train.py configs/recognition/aim/vitclip_utuner_base_sthv2.py

# python tools/train.py configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_hmdb51.py

# python tools/train.py configs/recognition/aim/vitclip_utuner_base_k400.py

# python tools/train.py configs/recognition/aim/vitclip_utuner_base_diving48.py

# python tools/train.py configs/recognition/aim/vitclip_zeroI2V_base_diving48.py

# python tools/train.py configs/recognition/aim/vitclip_tps_base_hmdb51.py

# python tools/train.py configs/recognition/aim/vitclip_vpt_base_hmdb51.py

# python tools/train.py configs/recognition/aim/vitclip_tome_base_hmdb51.py

# python tools/train.py configs/recognition/aim/vitclip_ats_base_hmdb51.py

python tools/train.py configs/recognition/aim/vitclip_utuner_base_hmdb51.py

# python tools/train.py configs/recognition/aim/vitclip_zeroI2V_base_hmdb51.py

# if [ $? -eq 0 ]; then
#     # 如果第一个脚本成功运行，则运行第二个 Python 脚本
#     python tools/train.py configs/recognition/aim/vitclip_zeroI2V_base_hmdb51.py
# else
#     # 如果第一个脚本失败，可以在此处添加处理失败情况的代码
#     echo "第一个脚本运行失败"
# fi

# PORT=29666 bash tools/dist_train.sh configs/recognition/aim/vitclip_flash_base_hmdb51.py 1 
# python tools/train.py configs/recognition/aim/vitclip_flash_base_hmdb51.py
