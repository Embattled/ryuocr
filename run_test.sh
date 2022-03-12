#! /bin/bash

cd /home/eugene/workspace/ryuocr
python3 ryuocr.py \
    --test \
    -c config/torch.yml \
    --model 2021-12-21-15-27-26torch \
    --best