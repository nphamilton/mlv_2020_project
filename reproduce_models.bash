#!/bin/bash

cd DDPG
python ddpg.py --random-seed 8 --log-path './rs_8'
python ddpg.py --random-seed 1964 --log-path './rs_1964'
python ddpg.py --random-seed 1754 --log-path './rs_1754'

cd ../DDPG-CBF
python ddpg.py --random-seed 8 --log-path './rs_8'
python ddpg.py --random-seed 1964 --log-path './rs_1964'
python ddpg.py --random-seed 1754 --log-path './rs_1754'

cd ..
python plot_results.py --folder1 'rs_8' --folder2 'rs_1964' --folder3 'rs_1754'
python generate_table.py --folder1 'rs_8' --folder2 'rs_1964' --folder3 'rs_1754'