import os

common_settings = ' ' + '--task sketching --dataset cheetah --save_particles --max_iter 30 --eval_interval 1 --particle_num 2000 --save_folder results --device cuda:3'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 0.3 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.5 --blur 0.01 --seed {}'.format(i) + common_settings)