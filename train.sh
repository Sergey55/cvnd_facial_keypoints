#!/bin/bash
python train.py --max_epochs 100 --gpus=1

gcloud compute instances stop $HOSTNAME