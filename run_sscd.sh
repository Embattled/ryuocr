#! /bin/bash

cd /home/eugene/workspace/ryuocr
python3 ryuocr.py --testsscd \
    -c config/testsscd.yml \
    --sscdrow 1 \
    --sscdcol 1 \
    --size 64 \
    --sscdmargin 0 \
    --sscdpath "/home/eugene/workspace/ryuocr/graph/sscd/a_PT_MO_CC_GF_RF_GN.png"
    # --sscdpath sscd_example.png
    # --size 64 \