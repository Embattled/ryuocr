#! /bin/bash

cd /home/eugene/workspace/ryuocr
python3 ryuocr.py \
    --oneoff \
    -c config/torch.yml \
    --valid_on_testset