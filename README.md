# Machine Learning enhanced Quantum Optimization on Parity architectures

This project includes important code used for my Master thesis. It should give 
researchers easy access to my project to replicate and understand results or to 
conduct further research on this topic.

## Description

Implementation of gate sequence optimization on parity architectures with the Proximal 
Policy Optimization (PPO) algorithm. There are three gate pools available or optimization. For more
information about the theory please refer to my thesis 
"Machine Learning enhanced Quantum Optimization on Parity architectures", which is included in the order "docs".

## Getting Started

### Dependencies

* should run on windows and linux
* includes a requirements.txt file 

### Installing

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Beat98/LHZ_QAOA_PPO.git

2. **Install required packages**:
    ```bash
   pip install -r requirements.txt
### Executing program

1. **Execute Run_LHZ_QAOA_PPO.py**:  
   Execute one PPO experiment. Multiple agents can be run after each other by 
   specifying num_executions. Recommended to gain accurate statistics.
    ```bash
   python Run_LHZ_QAOA_PPO.py
   ```
   You can modify parameters directly in the commandline. F.e.:
   ```bash
   python Run_LHZ_QAOA_PPO.py --n_max_gates 6 --num_executions 5
   ```
   Commandline arguments have priority over parameters from the custom configuration
   and over the defaults.  
   The data is saved in the data folder. Please specify a data_set_name.


2. **Get results with analyse_PPO_data.py**

   This function extracts useful statistics, prints them and plots the learning curves.
   Execute with:
   ```bash
   python analyse_PPO_data.py
   ```
   
3. **Run simulations in parallel on a linux cluster**
   
   You can run simulations with different parameter configurations in parallel using following sh template.
   Progress is stored in a txt file.
   ```bash
   sh run_sim_parallel_PPO.sh
   ```
   
4. **Make Environment changes**

   You can adjust f.e. the reward function in LHZ_QAOA_env.py. If you want to use a more recent QAOA or LHZ code
   here is the place to incorporate it.
   