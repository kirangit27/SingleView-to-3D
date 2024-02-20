# SingleView-to-3D

This project aims to examine the various loss types and decoder functions used for regression to voxels, point clouds, and mesh representation from a single-view RGB input.

## 0. Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing).
After unzipping, set the appropriate path references in the `dataset_location.py` file [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/dataset_location.py)

```
# It's better to do this after you've secured a GPU.
conda create -n pytorch3d-env python=3.9
conda activate pytorch3d-env
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install numpy PyMCubes matplotlib
```

Make sure you have installed the packages mentioned in `requirements.txt`.
This project will need the GPU version of pytorch.

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds, and meshes.

### 1.1. Fitting a voxel grid
In this subsection, binary cross entropy loss is defined that can help us <b>fit a 3D binary voxel grid</b>.
The loss functions are defined [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/losses.py#L5-L13) in `losses.py` file. 
Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 
#### Results - 
![Ground-Truth Voxel Grid](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_1_1-ground_truth_voxel.gif)  |  ![Predicted Voxel Grid](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_1_1-optimized_voxel.gif)
:-------------------------:|:-------------------------:
*Ground-Truth Voxel Grid*  |  *Predicted Voxel Grid*

### 1.2. Fitting a point cloud
In this subsection, the chamfer loss is defined which helps us <b> fit a 3D point cloud </b>.
The loss functions are defined [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/losses.py#L15-L25) in `losses.py` file.
Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 
#### Results - 
![Ground-Truth Point Cloud](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_1_2-ground_truth_pc.gif)  |  ![Optimized Point Cloud](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_1_2-optimized_pc.gif)
:-------------------------:|:-------------------------:
*Ground-Truth Point Cloud*  |  *Optimized Point Cloud*

### 1.3. Fitting a mesh
In this subsection, an additional smoothening loss is defined that can help us <b> fit a mesh</b>.
The loss functions are defined [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/losses.py#L27-L30) in `losses.py` file.
Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 
#### Results - 
![Ground-Truth Mesh](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_1_3-ground_truth_mesh.gif)  |  ![Optimized Mesh](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_1_3-optimized_mesh.gif)
:-------------------------:|:-------------------------:
*Ground-Truth Mesh*  |  *Optimized Mesh*
