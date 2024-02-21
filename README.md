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


## 2. Reconstructing 3D from single view
This section involves training a single-view image to the 3D pipeline for voxels, point clouds, and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

Pretrained ResNet18 features of images are provided to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and only use this if you are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument.

### 2.1. Image to voxel grid
In this subsection, a neural network is defined to decode binary voxel grids.
The decoder network is defined [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/model.py#L143-L153) in `model.py` file, and then referenced [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/model.py#L240-L245) in `model.py` file.

Run the file `python train_model.py --type 'vox'`, to train the single-view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After training, to visualize the input RGB, ground truth voxel grid, and predicted voxel in `eval_model.py` file use:
`python eval_model.py --type 'vox' --load_checkpoint`

#### Results - 
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_171_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_171_ground_truth.gif" alt="Ground-Truth Voxel Grid" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_171_prediction.gif" alt="Optimized Voxel Grid" width="300"/> |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_61_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_61_ground_truth.gif" alt="Ground-Truth Voxel Grid" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_61_prediction.gif" alt="Optimized Voxel Grid" width="300"/> |
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_5_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_5_ground_truth.gif" alt="Ground-Truth Voxel Grid" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_5_prediction.gif" alt="Optimized Voxel Grid" width="300"/> |
*Input RGB*  | *Ground-Truth Voxel Grid*  | *Optimized Voxel Grid*

### 2.2. Image to point cloud
In this subsection, the neural network is defined to decode point clouds.
Similar to above, the decoder network is defined [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/model.py#L155-L192) in `model.py` file, and then referenced [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/model.py#L247-L251) in `model.py` file

Run the file `python train_model.py --type 'point'`, to train the single-view to point-cloud pipeline, feel free to tune the hyperparameters as per your need.
After training, to visualize the input RGB, ground truth point cloud, and predicted point cloud in `eval_model.py` file use:
`python eval_model.py --type 'point' --load_checkpoint`

#### Results - 
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_35_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_35_ground_truth.gif" alt="Ground-Truth Point Cloud" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_35_prediction.gif" alt="Optimized Point Cloud" width="300"/> |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_19_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_19_ground_truth.gif" alt="Ground-Truth Point Cloud" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_19_prediction.gif" alt="Optimized Point Cloud" width="300"/> |
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_675_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_675_ground_truth.gif" alt="Ground-Truth Point Cloud" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_2_pc/point_675_prediction.gif" alt="Optimized Point Cloud" width="300"/> |
*Input RGB*  | *Ground-Truth Point Cloud*  | *Optimized Point Cloud*

### 2.3. Image to mesh
In this subsection, the neural network is defined  to decode mesh.
Similar to above, the decoder network is defined [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/model.py#L194-L224) in `model.py` file, and then referenced [here](https://github.com/kirangit27/SingleView-to-3D/blob/master/model.py#L253-L257) in `model.py` file

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need.

After training, to visualize the input RGB, ground truth mesh, and predicted mesh in `eval_model.py` file use:
`python eval_model.py --type 'mesh' --load_checkpoint`

#### Results - 
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_131_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_131_ground_truth.gif" alt="Ground-Truth Mesh" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_131_prediction.gif" alt="Optimized Mesh" width="300"/> |
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_13_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_13_ground_truth.gif" alt="Ground-Truth Mesh" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_13_prediction.gif" alt="Optimized Mesh" width="300"/> |
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_72_rgb_img.jpg" alt="Input RGB" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_72_ground_truth.gif" alt="Ground-Truth Mesh" width="300"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_72_prediction.gif" alt="Optimized Mesh" width="300"/> |
*Input RGB*  | *Ground-Truth Mesh*  | *Optimized Mesh*


### 2.4. Quantitative comparisons(10 points)
Quantitative comparisons of the F1 score of 3D reconstruction for meshes vs point cloud vs voxel grids.

For evaluation run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`

![Voxel F1 - 86.953](https://github.com/kirangit27/SingleView-to-3D/blob/master/evals/eval_vox.png) | ![Point Cloud F1 - 96.476](https://github.com/kirangit27/SingleView-to-3D/blob/master/evals/eval_point.png) | ![Mesh F1 - 88.182](https://github.com/kirangit27/SingleView-to-3D/blob/master/evals/eval_mesh.png)
:-------------------------:|:-------------------------:|:-------------------------:
*Voxel F1 - 86.953*  | *Point Cloud F1 - 96.476*  | *Mesh F1 - 88.182*

Quantitatively comparing the F1 score of 3D reconstruction for meshes, point clouds, and voxel grids involves evaluating how well each representation captures the details and structure of the 3D object being reconstructed. The F1 score is a metric that balances precision (the ratio of true positives to the sum of true positives and false positives) and recall (the ratio of true positives to the sum of true positives and false negatives).

- Meshes: Meshes generally provide high F1 scores as can be seen in the above plot. This is because meshes represent surfaces using a collection of vertices, edges, and faces, which can accurately capture the fine details of an object's surface.

- Point Clouds: Point clouds can achieve a good F1 score (best in our case), especially when they are densely sampled. However, it may struggle with capturing thin structures or surfaces that are not well-represented by the points.

- Voxel Grids: Voxel grids can provide a reasonable F1 score, particularly for objects that are well-suited to being represented in a grid-like structure. However, they may lose some fine details compared to meshes.


### 2.5. Analyse effects of hyperparameter variations
- Voxels: Varying the batch size. Increasing the batch size can have a positive impact on voxel prediction in 3D reconstruction tasks due to its effects on gradient noise reduction, improved generalization, better parallelization, reduced overfitting, a smoother loss landscape, and improved computational efficiency. However, it's important to note that increasing the batch size also comes with potential drawbacks, such as increased memory requirements and a possible reduction in convergence speed for certain types of models. Therefore, the choice of batch size should be carefully considered based on the specific characteristics of the dataset and the model being trained.

| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_2_rgb_img.jpg" alt="Input RGB" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_2_ground_truth.gif" alt="Ground-Truth Voxel Grid" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_2_prediction_old.gif" alt="Optimized Voxel Grid" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_2_prediction.gif" alt="Optimized Voxel Grid" width="225"/> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_4_rgb_img.jpg" alt="Input RGB" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_4_ground_truth.gif" alt="Ground-Truth Voxel Grid" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_4_prediction_old.gif" alt="Optimized Voxel Grid" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_1_vox/vox_4_prediction.gif" alt="Optimized Voxel Grid" width="225"/> |
*Input RGB*  | *Ground-Truth Voxel Grid*  | *Predicted voxel grid @ Batch Size 4* | *Predicted voxel grid @ Batch Size 16*

- Meshes: Varying num_workers. increasing the num_workers parameter in data loading for mesh prediction can lead to improved training efficiency by allowing for more parallelism in data processing. This can result in faster training times, more consistent GPU utilization, and potentially better-performing models. However, it's important to balance this with the capabilities of your hardware, as using too many workers can lead to resource contention and reduced performance.

| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_32_rgb_img.jpg" alt="Input RGB" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_32_ground_truth.gif" alt="Ground-Truth Mesh" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_32_prediction_old.gif" alt="Optimized Mesh" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_32_prediction.gif" alt="Optimized Mesh" width="225"/> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_112_rgb_img.jpg" alt="Input RGB" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_112_ground_truth.gif" alt="Ground-Truth Mesh" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_112_prediction_old.gif" alt="Optimized Mesh" width="225"/> | <img src="https://github.com/kirangit27/SingleView-to-3D/blob/master/results/Prob_2_3_mesh/mesh_112_prediction.gif" alt="Optimized Mesh" width="225"/> |
*Input RGB*  | *Ground-Truth Mesh*  | *Predicted Mesh @ num_workers 0* | *Predicted Mesh @ num_workers 4*



### 2.6. Model Interpretation 
To gain more gain insights from the results obtained I overlayed the predicted point clouds and meshes over the ground truth (used mesh ground truth for better visualization).

![Predicted point cloud overlayed on ground truth](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/pc_overlay.gif)|![Predicted Mesh overlayed on ground truth](https://github.com/kirangit27/SingleView-to-3D/blob/master/results/mesh_overlay.gif)
:-------------------------:|:-------------------------:
*Predicted point cloud overlayed on ground truth*  |  *Predicted Mesh overlayed on ground truth*


Overlaying prediction over ground truth serves several important purposes;
- Validation and Evaluation:
    - Quantitative Assessment: It allows for direct visual comparison between the predicted data and the ground truth. This aids in evaluating the accuracy of the prediction.
    - Error Analysis: By overlaying the predicted data over the ground truth, it becomes easy to identify areas where the prediction deviates from the actual data. This can provide insights into the strengths and weaknesses of the prediction algorithm.
- Visualization and Interpretation:
    - Visual Feedback: Overlaying helps in providing visual feedback to researchers, engineers, or practitioners. This aids in understanding the behavior of the algorithm and can lead to insights on how to improve it.
    - Error Localization: It aids in localizing specific areas or regions where the prediction is inaccurate. This information can guide further refinement or optimization of the algorithm.

