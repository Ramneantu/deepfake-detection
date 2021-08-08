https://github.com/marcotcr/lime
Unmasking DeepFakes with simple Features, Durall et al. (https://arxiv.org/abs/1911.00686)
http://www.niessnerlab.org/projects/roessler2019faceforensicspp.html

# Deep Fake Generation and Detection
## Overview
![Semantic description of image](overview_project.png)

In this project, we analyzed and recreated the main challenges in the task of Deep Fake Detection. The project setup can be seen in the figure above.

### Database
In order to analyze the abalitiy of generalizing to unseen images, we needed to have different datasets consisting real and fake images. The latter ones should best be attained by different methods. The idea is that deep fake detectors usually overfit on the generation methods that are given by the dataset. We therefore created 4 different datasets:
- c0 and c23: adapter from FaceForensics++ (http://www.niessnerlab.org/projects/roessler2019faceforensicspp.html)
