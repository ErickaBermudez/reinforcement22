conda activate lab
# change the followin line to indicate the name of the json to open
# each json is in its own folder
# example: vanilla_dqn_eps_greedy_cartpole_1000_spec
python run_lab.py slm_lab/spec/benchmark/dqn/dqn_cartpole.json vanilla_dqn_boltzmann_cartpole train # change to search if in the folder of search implementations 
