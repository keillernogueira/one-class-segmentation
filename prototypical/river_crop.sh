#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m1/ --model DenseNet121 --epoch_num 400 --margin 1 --miner True --weight_sampler False > coffee_crop/densenet/m1/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m2/ --model DenseNet121 --epoch_num 400 --margin 2 --miner True --weight_sampler False > coffee_crop/densenet/m2/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m3/ --model DenseNet121 --epoch_num 400 --margin 3 --miner True --weight_sampler False > coffee_crop/densenet/m3/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m4/ --model DenseNet121 --epoch_num 400 --margin 4 --miner True --weight_sampler False > coffee_crop/densenet/m4/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m5/ --model DenseNet121 --epoch_num 400 --margin 5 --miner True --weight_sampler False > coffee_crop/densenet/m5/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m7/ --model DenseNet121 --epoch_num 400 --margin 7 --miner True --weight_sampler False > coffee_crop/densenet/m7/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m8/ --model DenseNet121 --epoch_num 400 --margin 8 --miner True --weight_sampler False > coffee_crop/densenet/m8/out.txt

CUDA_VISIBLE_DEVICES=1 python learnt_prototypical_contrastive_main.py --operation Train --dataset Coffee_Crop --dataset_path /home/kno/datasets/coffee/montesanto_crop/ --training_images 0 1 --testing_images 2 --crop_size 128 --stride_crop 64 --batch_size 16 --output_path coffee_crop/densenet/m10/ --model DenseNet121 --epoch_num 400 --margin 10 --miner True --weight_sampler False > coffee_crop/densenet/m10/out.txt
