# Simple Embodied Language Learning as a Byproduct of Meta-Reinforcement Learning

This repo currently releases the environment studied in this paper.
The rest of the code will be uploaded soon, though already mostly exists publicly at this [GitHub repo](https://github.com/ezliu/dream).

## Introduction

Inspired by human language learning, we train meta-reinforcement learning agents to learn language as a byproduct of solving tasks.

[Evan Zheran Liu](https://ezliu.github.io/), [Sahaana Suri](https://cs.stanford.edu/~sahaana/), [Tong Mu](https://tongmu.github.io/), [Allan Zhou](https://bland.website/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/)\
International Conference on Machine Learning (ICML), 2023.

Also see our [paper](https://arxiv.org/abs/2306.08400).

## Requirements

This code requires Python3.6+.
The Python3 requirements are specified in `requirements.txt`.
We recommend creating a `virtualenv`, e.g.:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

The data for the environment is stored under `renders.tar.gz`, which needs to be fetched and placed into the current working directory from [this link](https://cs.stanford.edu/~evanliu/renders.tar.gz) and unpacked, e.g.:

```
wget https://cs.stanford.edu/~evanliu/renders.tar.gz
tar -xvf renders.tar.gz
```

## Usage

The main two environments can be found in `envs/minworld/office.py` and `envs/office.py`. They define environments with a Gym-like interface, which can be instantiated as follows:


```
from envs import office

seed = 0  # a random seed taking any integer value
env = office.MapMetaEnv.create_env(seed)  # a Gym environment
env.reset()
...
```

Similarly, a `MiniWorldOffice` environment can be created by appropriately modifying the above code.


## Citation

If you use this code, please cite our paper.

```
@article{liu2023simple,
  title={Simple Embodied Language Learning as a Byproduct of Meta-Reinforcement Learning},
  author={Liu, Evan Zheran and Suri, Sahaana and Mu, Tong and Zhou, Allan and Finn, Chelsea},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```
