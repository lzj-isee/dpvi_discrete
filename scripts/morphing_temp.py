import os


common_settings = ' ' + '--task morphing --save_particles --dataset cat.png[]thinspiral.png --hgradient --max_iter 20 --eval_interval 1 --max_src_num 1000 --max_tgt_num 2000 --save_folder results --device cuda:3 --plot_size 5'
for i in range(1):
    os.system('python3 main_iter.py --algorithm SD --lr 0.2 --blur 0.01 --seed {}'.format(i) + common_settings)