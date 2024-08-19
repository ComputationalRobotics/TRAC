# TRAC: Adaptive Parameter-free Optimization ⚡️
[![Open In Colab](https://img.shields.io/badge/project%20page-purple?logo=GitHub&logoColor=white
)](https://computationalrobotics.seas.harvard.edu/TRAC/)[![arXiv](https://img.shields.io/badge/arXiv-2405.16642-b31b1b.svg)](https://arxiv.org/abs/2405.16642) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c5OxMa5fiSVnl5w6J7flrjNUteUkp6BV?usp=sharing)[![Pypi](https://img.shields.io/badge/trac__optimizer-optimizer?logo=python&label=pip%20install&labelColor=white&color=gray&link=https%3A%2F%2Fpypi.org%2Fproject%2Ftrac-optimizer%2F)](https://pypi.org/project/trac-optimizer/)


This repository is the official implementation of the **TRAC** optimizer in ***Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning***.

How can you _quickly_ adapt to new tasks or distribution shifts? Without knowing when or how much to adapt? And without _ANY_ tuning? 
 🤔💭

Well, we suggest you get on the fast **TRAC** 🏎️💨.

**TRAC** is a parameter-free optimizer for continual environments inspired by [online convex optimization](https://arxiv.org/abs/1912.13213) and uses [discounted adaptive online prediction](https://arxiv.org/abs/2402.02720).

**Update [08/20/24]**: _TRAC is now supported for JAX and Optax!_

## Implement with only one line change with both PyTorch/JAX!
Like other [meta-tuners](https://openreview.net/pdf?id=uhKtQMn21D), TRAC can work with any of your continual, fine-tuning, or lifelong experiments with just one line change.
```python
pip install trac-optimizer
```
**PyTorch**
```python
from trac import start_trac
# original optimizer
optimizer = torch.optim.Adam
lr = 0.001
optimizer = start_trac(log_file='logs/trac.text', optimizer)(model.parameters(), lr=lr)
```
**JAX**
```python
from trac_optimizer.experimental.jax.trac import start_trac
# original optimizer
optimizer = optax.adam(1e-3)
optimizer = start_trac(optimizer)
```

After this modification, you can continue using your optimizer methods exactly as you did before. Whether it's calling `optimizer.step()` to update your model's parameters or `optimizer.zero_grad()` to clear gradients, everything stays the same. TRAC integrates into your existing workflow without any additional overhead.

## Control Experiments

We recommend running ``main.ipynb`` in Google Colab. This approach requires no setup, making it easy to get started with our control experiments. If you run locally, to install the necessary dependencies, simply:

```setup
pip install -r requirements.txt
```

## Vision-based RL Experiments

Our vision-based experiments for [Procgen](https://openai.com/index/procgen-benchmark/) and [Atari](https://www.gymlibrary.dev/environments/atari/index.html) are hosted in the `vision_exp` directory, which is based off [this Procgen Pytorch implementation](https://github.com/joonleesky/train-procgen-pytorch). 

To initiate an experiment with the default configuration in the Procgen "starpilot" environment, use the command below. You can easily switch to other game environments, like Atari, by altering the `--exp_name="atari"` parameter:

```bash
python vision_exp/train.py --exp_name="procgen" --env_name="starpilot" --optimizer="TRAC" --warmstart_step=0
```
