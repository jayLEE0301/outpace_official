# OUTPACE

This is a Pytorch implementation of **OUTPACE** from our paper: "Outcome-directed Reinforcement Learning by Uncertainty & Temporal Distance-Aware Curriculum Goal Generation" (ICLR 2023 Spotlight)

By [Daesol Cho*](https://daesolcho.github.io), [Seungjae Lee*](https://jaylee0301.github.io/) (*Equally contributed), and H. Jin Kim

A link to our paper can be found on arXiv(will be updated soon), and our project website can be found on [here](https://sjlee.cc/outpace).

<img width="100%" src="https://user-images.githubusercontent.com/30570922/215093920-a38380c2-d504-4dc5-9dd1-8b099cf51dfe.jpg"/>

## Setup Instructions
0. Create a conda environment:
```
conda env create -f outpace.yml
conda activate outpace
```

1. Add the necessary paths:
```
conda develop meta-nml
```

2. Install subfolder dependencies:
```
cd meta-nml && pip install -r requirements.txt
cd ..
chmod +x install.sh
./install.sh
```
3. Install [pytorch](https://pytorch.org/get-started/locally/) (use tested on pytorch 1.12.1 with CUDA 11.3)


4. Set config_path:
see config/paths/template.yaml

5. To run robot arm environment install [metaworld](https://github.com/rlworkgroup/metaworld):
```
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
```


## Usage
### Training and Evaluation

PointUMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointUMaze-v0 aim_disc_replay_buffer_capacity=10000 save_buffer=true adam_eps=0.01
```
PointNMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointNMaze-v0 aim_disc_replay_buffer_capacity=10000 adam_eps=0.01
```
PointSpiralMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 save_buffer=true aim_discriminator_cfg.lambda_coef=50
```
AntMazeSmall-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=AntMazeSmall-v0 aim_disc_replay_buffer_capacity=50000
```
sawyer_peg_pick_and_place
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=sawyer_peg_pick_and_place aim_disc_replay_buffer_capacity=30000 normalize_nml_obs=true normalize_f_obs=false normalize_rl_obs=false adam_eps=0.01
```
sawyer_peg_push
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=sawyer_peg_push aim_disc_replay_buffer_capacity=30000 normalize_nml_obs=true normalize_f_obs=false normalize_rl_obs=false adam_eps=0.01 hgg_kwargs.match_sampler_kwargs.hgg_L=0.5
```

Our code sourced and modified from official implementation of [MURAL](https://github.com/kevintli/mural), [AIM](https://github.com/iDurugkar/adversarial-intrinsic-motivation), and [HGG](https://github.com/Stilwell-Git/Hindsight-Goal-Generation) Algorithm. Also, we utilize [mujoco-maze](https://github.com/kngwyu/mujoco-maze) and [metaworld](https://github.com/rlworkgroup/metaworld) to validate our proposed method.



## Citation
If you use this repo in your research, please consider citing the paper as follows.
```
@inproceedings{choandlee2023outcome,
  title={Outcome-directed Reinforcement Learning by Uncertainty \& Temporal Distance-Aware Curriculum Goal Generation},
  author={Cho, Daesol and Lee, Seungjae and Kim, H Jin},
  booktitle={Proceedings of International Conference on Learning Representations},
  pages={},
  year={2023},
  organization={}
}
```
