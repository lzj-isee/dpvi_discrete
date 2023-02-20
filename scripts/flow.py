import os

common_settings = ' ' + '--task flow --dataset cifar10 --particle_num 1500 --tgt_num 1000 --max_iter 100 --eval_interval 5 --scaling 0.8 --blur 0.02 --power 1 --save_folder results --device cuda:3'
for i in range(1):
    # os.system('python3 main_iter.py --algorithm SD --lr 1.0 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.012 --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.014 --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.016 --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.018 --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.008 --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.006 --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.004 --noise_amp 0.003 --seed {}'.format(i) + common_settings)


    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.01 --al_warmup --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.01 --noise_amp 0.003 --seed {}'.format(i) + common_settings)

    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.02 --al_warmup --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.02 --noise_amp 0.003 --seed {}'.format(i) + common_settings)

    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.04 --al_warmup --noise_amp 0.003 --seed {}'.format(i) + common_settings)
    # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.04 --noise_amp 0.003 --seed {}'.format(i) + common_settings)

# common_settings = ' ' + '--task flow --dataset cifar10 --particle_num 3000 --tgt_num 2000 --max_iter 100 --eval_interval 5 --scaling 0.95 --blur 0.02 --power 2 --save_folder results --device cuda:3'
# for i in range(1):
#     os.system('python3 main_iter.py --algorithm SD --lr 0.3 --seed {}'.format(i) + common_settings)
#     # os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.03 --al_warmup --noise_amp 0.003 --seed {}'.format(i) + common_settings)