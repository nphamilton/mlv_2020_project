#!/bin/bash

cd DDPG
python ddpg.py --random-seed 8 --log-path './rs_8'
python ddpg.py --random-seed 1964 --log-path './rs_1964'
python ddpg.py --random-seed 1754 --log-path './rs_1754'

python ddpg.py --random-seed 8 --constrain 1 --log-path './constrained_rs_8'
python ddpg.py --random-seed 1964 --constrain 1 --log-path './constrained_rs_1964'
python ddpg.py --random-seed 1754 --constrain 1 --log-path './constrained_rs_1754'

cd ../DDPG-CBF
python ddpg.py --random-seed 8 --log-path './rs_8'
python ddpg.py --random-seed 1964 --log-path './rs_1964'
python ddpg.py --random-seed 1754 --log-path './rs_1754'

cd ..
python plot_results.py
python generate_table.py