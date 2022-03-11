CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 0,1 python -m main \
--dir '/media/new-data/Data-0.14-0.94-mutual' \
--patch_num 'repeated' \
--lr 0.005 \
--num_iter 1000 \
--indep_thresh 0.14 \
--accu_thresh 0.94 \
--var_thresh 0.0001 \
--image_num 1000 \
