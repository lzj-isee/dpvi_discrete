import os

common_settings = ' ' + '--task color_transfer --src_downscale 2 --tgt_downscale 8 --square_size 512 --dataset group3 --max_iter 20 --eval_interval 1 --save_folder results --device cuda:3'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 1.0 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.03 --al_warmup --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --save_particles --algorithm SDCA --lr 0.3 --alpha 10.0 --seed {}'.format(i) + common_settings)