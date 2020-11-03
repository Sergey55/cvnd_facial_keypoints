#!/bin/bash
python train.py --max_epochs 200 --gpus=1

gcloud compute instances stop $HOSTNAME