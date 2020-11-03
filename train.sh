#!/bin/bash
python train.py --max_epochs 150 --gpus=1

gcloud compute instances stop $HOSTNAME