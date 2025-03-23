# Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning

Here we provide sourcecodes of RLDE-AFL, which has been recently accpeted by GECCO 2025.

## Citation

The PDF version of the paper is available [here](). If you find our RLDE-AFL useful, please cite it in your publications or projects.

```latex
@inproceedings{guo2025rldeafl,
  title={Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning},
  author={Guo, Hongshu and Ma, Sijie and Huang, Zechuan and Hu, Yuzhi and Ma, Zeyuan and Zhang, Xinglin and Gong, Yue-Jiao},
  booktitle={},
  year={2025}
}
```

## Requirements
You can install all of dependencies of RLDE-AFL via the command below.
```bash
pip install -r requirements.txt
```

## Train
The training process can be activated via the command below, which is just an example.
```bash
python main.py --run_experiments --problem bbob --difficulty difficult --device cuda --max_epoch 24 --pop_size 100 --max_fes 20000 --crossover_op binomial exponential MDE_pBX --reward_ratio 1 --seed 7 --trainset_seed 13 --testset_seed 1024 --rollout_interval 10  --fe_train --run_name test_run
```
For more adjustable settings, please refer to `main.py` and `config.py` for details.

Recording results: Log files will be saved to `./outputs/logs/train/` . The saved checkpoints will be saved to `./outputs/model/train/`. TensorBoard files will be located in `./tensorboard/`. The file structure is as follow:
```
outputs
|--logs
   |--train
      |--run_name
         |--...
|--models
   |--train
      |--run_name
         |--Epoch
            |--epoch1.pkl
            |--epoch2.pkl
            |--...
```

## Rollout
The rollout process can be easily activated via the command below.
```bash
python main.py --test --problem bbob --difficulty difficult --device cuda --max_epoch 100 --pop_size 100 --max_fes 2000 --crossover_op binomial exponential MDE_pBX --reward_ratio 1 --seed 7 --trainset_seed 13 --testset_seed 1024 --rollout_interval 10  --fe_train --run_name test_run --agent_load_dir [The checkpoint saving directory] --agent_model [The purpose model name]; 
```
To use the test_model.pkl file located in the home directory as the target model, you can modify the command as follows:
```bash
python main.py --test --problem bbob --difficulty difficult --device cuda --max_epoch 100 --pop_size 100 --max_fes 2000 --crossover_op binomial exponential MDE_pBX --reward_ratio 1 --seed 7 --trainset_seed 13 --testset_seed 1024 --rollout_interval 10  --fe_train --run_name test_run --agent_load_dir ./ --agent_model test_model
```