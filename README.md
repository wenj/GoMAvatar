# GoMAvatar: Efficient Animatable Human Modeling from Monocular Video Using Gaussians-on-Mesh

CVPR 2024

[Paper](https://arxiv.org/abs/2404.07991) | [Project Page](https://wenj.github.io/GoMAvatar/)

```bibtex
@inproceedings{wen2024gomavatar,
    title={{GoMAvatar: Efficient Animatable Human Modeling from Monocular Video Using Gaussians-on-Mesh}},
    author={Jing Wen and Xiaoming Zhao and Zhongzheng Ren and Alex Schwing and Shenlong Wang},
    booktitle={CVPR},
    year={2024}
}
```

## Requirements

Our codes are tested in
* CUDA 11.6
* PyTorch 1.13.0
* PyTorch3D 0.7.0

Install the required packages:
```Shell
conda create -n GoMAvatar
conda activate GoMAvatar

conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt

# install pytorch3d 
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# install gaussian splatting
pip install git+"https://github.com/graphdeco-inria/diff-gaussian-rasterization"
```

## Data preparation
### Prerequisites
Download SMPL v1.0.0 models from [here](https://smpl.is.tue.mpg.de/download.php) and put the `.pkl` files under `utils/smpl/models`.
You may need to remove the Chumpy objects following [here](https://github.com/vchoutas/smplx/tree/main/tools).

### ZJU-MoCap

First download the [ZJU-MoCap](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) dataset and save the raw data under `data/zju-mocap`.

Run the following script to preprocess the dataset:
```Shell
cd scripts/prepare_zju-mocap
python prepare_dataset.py --cfg "$SCENE".yaml
```
Change `$SCENE` to one of 377, 386, 387, 392, 393, 394.

The folder will be in the following structure:
```Shell
├── data
    ├── zju-mocap
        ├── 377
        ├── 386
        ├── ...
        ├── CoreView_377
        ├── CoreView_386
        ├── ...
```
Folders named after scene ID only are preprocessed training data, while those prefixed with `CoreView_` are raw data. 

### PeopleSnapshot

Download the [PeopleSnapshot](https://graphics.tu-bs.de/people-snapshot) dataset and save the files under `data/snapshot`.

Download the refined training poses from [here](https://github.com/tijiang13/InstantAvatar/tree/master/data/PeopleSnapshot). 

Run the following script to preprocess the training and test set.
```Shell
cd scripts/prepare_snapshot
python prepare_dataset.py --cfg "$SCENE".yaml # training set
python prepare_dataset.py --cfg "$SCENE"_test.yaml # test set
```
`$SCENE` is one of `female-3-casual`, `female-4-casual`, `male-3-casual` and `male-4-casual`.

After the preprocessing, the folder will be in the following structure:
```Shell
├── data
    ├── snapshot
        ├── f3c_train
        ├── f3c_test
        ├── f4c_train
        ├── f4c_test
        ├── ...
        ├── female-3-casual
        ├── female-4-casual
        ├── ...
        ├── poses # refined training poses
            ├── female-3-casual
                ├── poses
                    ├── anim_nerf_test.npz
                    ├── anim_nerf_train.npz
                    ├── anim_nerf_val.npz
            ├── ...
        
```
Folders ended with `_train` or `_test` are preprocessed data.

## Rendering and evaluation

We provide the pretrained checkpoints in this [link](https://uofi.box.com/s/onwfp29ej03sr2ci7mm59nu74v6i0ip3). To reproduce the rendering results in the paper, run
```Shell
# ZJU-MoCap novel view synthesis
python eval.py --cfg exps/zju-mocap_"$SCENE".yaml --type view

# ZJU-MoCap novel pose synthesis
python eval.py --cfg exps/zju-mocap_"$SCENE".yaml --type pose
```
On the PeopleSnapshot dataset, we follow [Anim-NeRF](https://github.com/JanaldoChen/Anim-NeRF) and [InstantAvatar](https://github.com/tijiang13/InstantAvatar) to refine test poses:
```Shell
python train_pose.py --cfg exps/snapshot_"$SCENE".yaml
```
Please check `exps/` for detailed configuration files.

You can run 360 degree freeview rendering using the following command
```Shell
python eval.py --cfg exps/zju-mocap_"$SCENE".yaml --type freeview
```
Use `--frame_idx` to specify the training frame id and `--n_frames` to set the number of views.

Or you can render novel poses from [MDM](https://guytevet.github.io/mdm-page/):
```Shell
python eval.py --cfg exps/zju-mocap_"$SCENE".yaml --type pose_mdm --pose_path data/mdm_poses/sample.npy
```
We provide an example of pose trajectory in `data/pose_mdm/sample.npy`.

## Training

Run the following command to train from scratch:
```Shell
# ZJU-MoCap
python train.py --cfg exps/zju-mocap_"$SCENE".yaml

# PeopleSnapshot
python train.py --cfg exps/snapshot_"$SCENE".yaml
```

## Acknowledgements

This project builds upon [HumanNeRF](https://github.com/chungyiweng/humannerf) and [MonoHuman](https://github.com/Yzmblog/MonoHuman/tree/main). We appreciate the authors for their great work!
