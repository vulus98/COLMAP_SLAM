# COLMAP-SLAM
[3D Vision](https://www.cvg.ethz.ch/teaching/3dvision/) project supervised by [Paul-Edouard](https://github.com/Skydes)


## Abstract
State-of-the-art Structure from Motion algorithms such as [COLMAP](https://github.com/colmap/colmap) are highly robust in reconstruction but are slow and often don't scale well. This makes them unsuitable for long video data. On the other hand, SLAM systems can process videos (sequential images) in real-time but fall behind COLMAP in map quality. The goal of this project is to combine the best of both worlds to obtain a fast, robust and scalable SLAM system. We demonstrate that we partially achieved our goals by utilizing components of COLMAP and ideas from [ORB-SLAM](https://github.com/raulmur/ORB_SLAM). The quality of our map, however, is not yet comparable with the state-of-the-art.

## Installation
We strongly recommend using Linux. It has been tested out on Ubuntu 20.04.1 LTS.

1. Install [COLMAP](https://colmap.github.io/)
2. Install [pycolmap](https://github.com/colmap/pycolmap)
3. Install [hloc](https://github.com/cvg/Hierarchical-Localization)
4. Install [pyceres](https://github.com/cvg/pyceres)

## APP
// TODO: How to run and use the app

## General Pipeline

![pipeline](https://user-images.githubusercontent.com/17593719/173319164-a413e146-b05c-4326-a851-a81927f35f8a.png)
