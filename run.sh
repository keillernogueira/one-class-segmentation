#!/bin/bash

python main.py --operation Train --output_path outputs/ --dataset_path /mnt/DADOS_PARIS_1/keiller/datasets/rios/ --training_images no so --testing_images se --crop_size 64 --stride_crop 50 --model WideResNet --batch_size 16 --epoch_num 500 > outputs/out.txt

python main.py --operation Plot --output_path outputs/ --dataset_path /mnt/DADOS_PARIS_1/keiller/datasets/rios/ --training_images no so --testing_images se --crop_size 64 --stride_crop 64 --model WideResNet --batch_size 32
