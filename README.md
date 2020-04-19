# mlv_2020_project
A quick dive into comparing the verifiable safety of safe reinforcement learning vs state-of-the-art deep reinforcement 
learning.

## Installation and Reproducability
To install and reproduce the results presented in the report, you will need 
[Anaconda Python](https://www.anaconda.com/distribution/) installed (make sure you have the Python3.7 version) and 
access to a bash-anabled terminal.

Once you have Anaconda installed, navigate in your terminal to this directory and run 

```bash
./setup.bash
```

Now your system is setup and can run the reproduce code:
```bash
./reproduce_models.bash
```
Feel free to make modifications within this bash script to try different random seeds. Note: this will overwrite the 
models currently available in the repo.

To verify the models, make sure you have the [NNV]() tool installed and setup on your system. Then, add `verify.m` to 
your MATLAB working directory and run it. That will run all of the verification tests described in the report.

If there are any issues using or running the files in this repo, please contact me at nathaniel.p.hamilton@vanderbilt.edu
and I'll try to help you as best I can.

## Control Barrier Functions

## DDPG
