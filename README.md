# Dynamics Learning while Retaining Memories

----

Original PyTorch implementation of **DRAGO** used on TD-MPC


<p align="center">
  <br><img src='media/drago.png' width="600"/><br>
   <a href="https://github.com/yixiang-sun/Knowledge-Retention-for-Continual-MBRL/">[Website]</a>
</p>


## Method

**DRAGO** is a framework for knowledge retention in continuel model-based reinforcement 
learning. By generating previously seen transitions and tracing back to previously familiar
states, DRAGO is able to enhance knowledge retention of dynamics in MBRL algorithms,
resulting in a more general world model that improves knowledge transfer in learning
new tasks.



## Instructions

Assuming that you already have [MuJoCo](http://www.mujoco.org) installed, install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate drago
```

After installing dependencies, you can continually learn a world model and corresponding
TD-MPC models of a sequence of tasks (in this case, [cheetah-run, cheetah-jump, cheetah-run-backwards]): 

```
python src/train.py env=dmcontrol domain=cheetah
```

Evaluation videos and model weights can be saved with arguments `save_video=True` and `save_model=True`. Refer to the `cfgs` directory for a full list of options and default hyperparameters, and see `tasks.txt` for a list of supported tasks. We also provide results for all 23 state-based DMControl tasks in the `results` directory.

The training script supports both local logging as well as cloud-based logging with [Weights & Biases](https://wandb.ai). To use W&B, provide a key by setting the environment variable `WANDB_API_KEY=<YOUR_KEY>` and add your W&B project and entity details to `cfgs/default.yaml`.