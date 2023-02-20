import os

common_settings = ' ' + '--task sketching --dataset cheetah --save_particles --max_time 360 --eval_interval 10 --particle_num 2000 --save_folder results_save/sketching_time --device cuda:3'
for i in range(10):
    # os.system('python3 main_time.py --algorithm SD --lr 1.0 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_time.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_time.py --algorithm SDDK --lr 1.0 --alpha 0.5 --blur 0.01 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_time.py --algorithm MMDF --lr 0.03 --beta 0.0 --beta_min 0.001 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    # os.system('python3 main_time.py --algorithm MMDFCA --lr 0.03 --alpha 5.0 --beta 0.0 --beta_min 0.001 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_time.py --algorithm MMDFDK --lr 0.03 --al_warmup --alpha 0.5 --noise_amp 0.01 --beta 0.0 --beta_min 0.001 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    