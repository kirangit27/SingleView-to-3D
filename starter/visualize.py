
import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from PIL import Image, ImageDraw
from starter.utils import *
import imageio
import mcubes

device = get_device()

def render_vox_360(vox, vox_size=32, num_views=18, output_path="results", image_size=512, device=None):

    bin = np.squeeze(vox, axis=0)
    vertices, faces = mcubes.marching_cubes(bin.squeeze().detach().cpu().numpy(), isovalue=0.0)
    dmin, dmax  = -1.1, 1.1

    vertices = torch.tensor(vertices).float()
    vertices = (((vertices/vox_size) * (dmax - dmin)) + dmin).unsqueeze(0)
    faces = (torch.tensor(faces.astype(int))).unsqueeze(0)

    z = vertices[:,:, 2]
    z_max, z_min = torch.max(z), torch.min(z)

    alpha = ((z - z_min) / (z_max - z_min)).to(device=device)
    alpha = alpha.unsqueeze(2).expand(-1, -1, 3)
    textures = torch.ones_like(vertices,device=device)

    color_1 = torch.tensor([1, 0, 0])
    color_2 = torch.tensor([0, 0, 1])
    color_1_tex = textures * torch.tensor(color_1,device=device)
    color_2_tex =textures * torch.tensor(color_2,device=device)
    color = (1 - alpha) * color_1_tex + alpha * color_2_tex
    textures = pytorch3d.renderer.TexturesVertex(color)

    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures).to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    azim = torch.linspace(0, 360, num_views)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=azim)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras( R=R, T=T, device=device)

    images = []
    for k in cameras:
        image = renderer(mesh, cameras=k)
        image = image.cpu().numpy()[0, ..., :3]
        images.append(image)

    images = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
    imageio.mimsave(output_path, images, duration=60, loop=0)
    print("Done!")



def render_points_360(point_cloud,num_views=18, output_path="results", image_size=512, device=None):

    vertices = point_cloud
    rgb = (vertices-vertices.min())/(vertices.max()-vertices.min())

    color_1 = torch.tensor([1, 0, 0])
    color_2 = torch.tensor([0, 0, 1])
    color_1 = color_1.to(device).to(device)
    color_2 = color_2.to(device).to(device)
    color = ((1 - rgb[:, :, None])*color_1) + (rgb[:, :, None]*color_2)
    color = color.squeeze(0).permute(1,0,2)

    pc = pytorch3d.structures.Pointclouds(points=point_cloud,features=color).to(device)
    renderer = get_points_renderer(image_size=image_size, background_color=(1, 1, 1) )

    azim = torch.linspace(0, 360, num_views)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=1, elev=0, azim=azim)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    images=[]
    for k in cameras:
        image = (renderer(pc, cameras=k))
        image = image.detach().cpu().numpy()[0, ..., :3]
        images.append(image)

    images = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
    imageio.mimsave(output_path, images, duration=60, loop=0)
    print("Done!")



def render_mesh_360(mesh,num_views=18, output_path="results", image_size=512, device=None):

    vertices = mesh.verts_packed().unsqueeze(0)
    z = vertices[:,:, 2]
    z_max, z_min = torch.max(z), torch.min(z)

    alpha = ((z - z_min) / (z_max - z_min)).to(device=device)
    alpha = alpha.unsqueeze(2).expand(-1, -1, 3)
    textures = torch.ones_like(vertices,device=device)
    color_1 = [1, 0, 0]
    color_2 = [0, 0, 1]
    color_1_tex = textures * torch.tensor(color_1,device=device)
    color_2_tex =textures * torch.tensor(color_2,device=device)
    color = (1 - alpha) * color_1_tex + alpha * color_2_tex

    renderer = get_mesh_renderer(device=device)
    mesh.textures = pytorch3d.renderer.TexturesVertex(color)

    azim = torch.linspace(0, 360, num_views)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=1, elev=0, azim=azim)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images=[]
    for i in cameras:
        img=(renderer(mesh.to(device),device=device, cameras=i, lights=lights))
        img=img.detach().cpu().numpy()[0, ..., :3]
        images.append(img)

    images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    imageio.mimsave(output_path, images, duration=60, loop=0)
    print("Done!")
