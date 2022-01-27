
CUDA_VISIBLE_DEVICES=1 python main_baseline.py --operation Train --output_path coffee/baseline/1/ --dataset Coffee --dataset_path /home/kno/datasets/coffee/ --training_images 2 3 4 5 --testing_images 1 --crop_size 64 --stride_crop 50 --model WideResNet --batch_size 16 --epoch_num 500 > coffee/baseline/1/out.txt

CUDA_VISIBLE_DEVICES=1 python main_baseline.py --operation Train --output_path coffee/baseline/2/ --dataset Coffee --dataset_path /home/kno/datasets/coffee/ --training_images 1 3 4 5 --testing_images 2 --crop_size 64 --stride_crop 50 --model WideResNet --batch_size 16 --epoch_num 500 > coffee/baseline/2/out.txt

CUDA_VISIBLE_DEVICES=1 python main_baseline.py --operation Train --output_path coffee/baseline/3/ --dataset Coffee --dataset_path /home/kno/datasets/coffee/ --training_images 1 2 4 5 --testing_images 3 --crop_size 64 --stride_crop 50 --model WideResNet --batch_size 16 --epoch_num 500 > coffee/baseline/3/out.txt

CUDA_VISIBLE_DEVICES=1 python main_baseline.py --operation Train --output_path coffee/baseline/4/ --dataset Coffee --dataset_path /home/kno/datasets/coffee/ --training_images 1 2 3 5 --testing_images 4 --crop_size 64 --stride_crop 50 --model WideResNet --batch_size 16 --epoch_num 500 > coffee/baseline/4/out.txt

CUDA_VISIBLE_DEVICES=1 python main_baseline.py --operation Train --output_path coffee/baseline/5/ --dataset Coffee --dataset_path /home/kno/datasets/coffee/ --training_images 1 2 3 4 --testing_images 5 --crop_size 64 --stride_crop 50 --model WideResNet --batch_size 16 --epoch_num 500 > coffee/baseline/5/out.txt