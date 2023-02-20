import os
# -------------------------------------------------------------------- 1000 particles ----------------------------------------------------------------------
common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 15000 --eval_interval 1000 --max_src_num 1000 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_1000 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm MMDF --lr 0.03 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFCA --lr 0.03 --alpha 5.0 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFDK --lr 0.03 --al_warmup --alpha 0.3 --noise_amp 0.01 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 300 --eval_interval 20 --max_src_num 1000 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_1000 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 1.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.5 --blur 0.01 --seed {}'.format(i) + common_settings)

# -------------------------------------------------------------------- 1500 particles ----------------------------------------------------------------------
common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 15000 --eval_interval 1000 --max_src_num 1500 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_1500 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm MMDF --lr 0.03 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFCA --lr 0.03 --alpha 5.0 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFDK --lr 0.03 --al_warmup --alpha 0.3 --noise_amp 0.01 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 300 --eval_interval 20 --max_src_num 1500 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_1500 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 1.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.5 --blur 0.01 --seed {}'.format(i) + common_settings)


# -------------------------------------------------------------------- 2000 particles ----------------------------------------------------------------------

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 15000 --eval_interval 1000 --max_src_num 2000 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_2000 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm MMDF --lr 0.03 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFCA --lr 0.03 --alpha 5.0 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFDK --lr 0.03 --al_warmup --alpha 0.3 --noise_amp 0.01 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 300 --eval_interval 20 --max_src_num 2000 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_2000 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 1.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.5 --blur 0.01 --seed {}'.format(i) + common_settings)

# -------------------------------------------------------------------- 500 particles ----------------------------------------------------------------------

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 15000 --eval_interval 1000 --max_src_num 500 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_500 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm MMDF --lr 0.01 --beta 0.01 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFCA --lr 0.01 --alpha 5.0 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFDK --lr 0.01 --al_warmup --alpha 0.1 --noise_amp 0.01 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 300 --eval_interval 20 --max_src_num 500 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_500 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 1.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.3 --blur 0.01 --seed {}'.format(i) + common_settings)

# -------------------------------------------------------------------- 250 particles ----------------------------------------------------------------------
common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 15000 --eval_interval 1000 --max_src_num 250 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_250 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm MMDF --lr 0.005 --beta 0.01 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFCA --lr 0.005 --alpha 5.0 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm MMDFDK --lr 0.005 --al_warmup --alpha 0.1 --noise_amp 0.01 --beta 0.0 --beta_min 0.01 --bwType nei --knType imq  --seed {}'.format(i) + common_settings)

common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 300 --eval_interval 20 --max_src_num 250 --max_tgt_num 2000 --save_folder results/cat[]thinspiral_250 --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 1.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDCA --lr 1.0 --alpha 10.0 --blur 0.01 --seed {}'.format(i) + common_settings)
    os.system('python3 main_iter.py --algorithm SDDK --lr 1.0 --alpha 0.1 --blur 0.01 --seed {}'.format(i) + common_settings)
