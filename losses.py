import torch.nn as N
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d

	# implement some loss for binary voxel grids
    BCE_loss = N.BCEWithLogitsLoss()
    loss = BCE_loss(voxel_src,voxel_tgt)
    return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3

	# implement chamfer loss from scratch
    knn1 = knn_points(point_cloud_src, point_cloud_tgt)
    knn2 = knn_points(point_cloud_tgt, point_cloud_src)

    chamfer_x = knn1.dists.squeeze(0)
    chamfer_y = knn2.dists.squeeze(0)
    loss_chamfer = (chamfer_x + chamfer_y).mean()
    return loss_chamfer

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss
    loss_laplacian = mesh_laplacian_smoothing(mesh_src)
    return loss_laplacian






