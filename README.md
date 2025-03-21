# DISTRAL_DOOM

<p align="center">
  <video width="400" controls>
  <source src="env2-Checking multiInp.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

  <img src="llustration-of-the-Distral-framework.png" width="400" />
</p>

## Project Description
This project implements the DISTRAL (Distill & Transfer Learning) algorithm [1] in VizDoom environments [2]. DISTRAL is a multi-task reinforcement learning algorithm that learns a shared policy across multiple tasks while allowing task-specific policies to adapt to their individual environments.

### VizDoom
VizDoom [2] is a Doom-based AI research platform that allows agents to learn from visual information. It provides various scenarios where agents must learn to navigate 3D environments and perform actions like moving, shooting, and collecting items.

### DISTRAL Algorithm
DISTRAL [1] (Distill & Transfer Learning) works by:
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

## Citation
If you use this code in your research, please cite:
```bibtex
@misc{dennis2024distral,
  author = {Dennis, Cameron David Barrie and Dantchev, Stefan},
  title = {DISTRAL Implementation in VizDoom Environments},
  year = {2024},
  publisher = {Durham University},
  howpublished = {\url{https://github.com/DennisC24/DISTRAL_DOOM}}
}
```

## References

[1] Teh, Y. W., Bapst, V., Czarnecki, W. M., Quan, J., Kirkpatrick, J., Hadsell, R., Heess, N., & Pascanu, R. (2017). Distral: Robust multitask reinforcement learning. Advances in Neural Information Processing Systems, 30.

[2] Kempka, M., Wydmuch, M., Runc, G., Toczek, J., & Jaśkowski, W. (2016). ViZDoom: A Doom-based AI research platform for visual reinforcement learning. IEEE Conference on Computational Intelligence and Games, 1-8.
