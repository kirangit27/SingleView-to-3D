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
In this subsection, we will define binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b>.
Define the loss functions [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/losses.py#L4-L9) in `losses.py` file. 
For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

**Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.**

### 1.2. Fitting a point cloud
In this subsection, we will define chamfer loss that can help us <b> fit a 3D point cloud </b>.
Define the loss functions [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/losses.py#L11-L15) in `losses.py` file.
<b>We expect you to write your code for this and not use any pytorch3d utilities. You are allowed to use functions inside pytorch3d.ops.knn such as knn_gather or knn_points</b>

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

**Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.**

### 1.3. Fitting a mesh
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.
Define the loss functions [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/losses.py#L17-L20) in `losses.py` file.

For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 
