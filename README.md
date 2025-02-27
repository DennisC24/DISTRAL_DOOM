# DISTRAL_DOOM

<p align="center">
  <img src="DISTRAL.png" width="400" />
  <img src="doom.gif" width="400" />
</p>

## Project Description
This project implements the DISTRAL (Distill & Transfer Learning) algorithm in VizDoom environments. DISTRAL is a multi-task reinforcement learning algorithm that learns a shared policy across multiple tasks while allowing task-specific policies to adapt to their individual environments.

### VizDoom
VizDoom is a Doom-based AI research platform that allows agents to learn from visual information. It provides various scenarios where agents must learn to navigate 3D environments and perform actions like moving, shooting, and collecting items.

### DISTRAL Algorithm
DISTRAL (Distill & Transfer Learning) works by:
1. Learning a shared policy across all tasks
2. Training task-specific policies that are regularized towards the shared policy
3. Using KL divergence to balance between task-specific optimization and policy sharing
4. Enabling knowledge transfer between tasks through the shared policy

## Dependencies
Required Python packages and versions:
```
gymnasium==1.0.0
numpy==1.24.4
opencv-python==4.11.0.86
optuna==4.2.1
pandas==2.0.3
torch==2.1.1
torchvision==0.16.1
vizdoom==1.2.4
```

To install all dependencies:
```
pip install -r requirements.txt
```
