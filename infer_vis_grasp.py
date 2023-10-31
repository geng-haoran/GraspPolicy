import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup
sys.path.append(sys.path[0] + '/graspness_implementation')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspness_implementation.models.graspnet import GraspNet, pred_decode
from graspness_implementation.dataset.graspnet_dataset import minkowski_collate_fn
from graspness_implementation.utils.collision_detector import ModelFreeCollisionDetector
from graspness_implementation.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

from utils.config_parser import parse_yaml

def data_process_asgrasp():
    cloud_sampled = np.array(o3d.io.read_point_cloud("graspness_implementation/test_real/pcd_0.ply").points)
    data_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs["voxel_size"],
                'feats': np.ones_like(cloud_sampled).astype(np.float32)}
    return data_dict

def data_process_gapartnet():
    """ rgbd from gapartnet """
    # depth = np.load("/data/songlin/projects/LLM-GAPartNet/GAPartNet_self/dataset/render_tools/dump_save_path/41083/20b4d0/depth_part.npz")['depth_map']
    # rgb  = np.ascontiguousarray(Image.open("/data/songlin/projects/LLM-GAPartNet/GAPartNet_self/dataset/render_tools/dump_save_path/41083/20b4d0/rgb_2.png"))
    # height, width = depth.shape[:2]
    # camera_instrinisc = np.array([965.6854248046875, 965.6854248046875, 400.0, 400.0])

    """ rgbd from kinect_sub """
    rgb = np.load("test_real/rgb_4.npy")
    rgb = rgb[...,[2,1,0]].copy()
    depth = np.load("test_real/depth_4.npy")
    depth[depth > 800] = 0.
    depth = depth * 1e-3
    camera_instrinisc = np.array([973.9103393554688, 973.8275756835938, 1022.8403930664062, 782.8876342773438])
    height, width = depth.shape[:2]

    if False:
        """ visualize original rgb-d pointcloud """
        depth_raw = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        color_raw = o3d.geometry.Image(np.ascontiguousarray(rgb).astype(np.uint8))
        rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 
                depth_scale=1., depth_trunc=10, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 
                    *camera_instrinisc)
        pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_raw, intrinsic)
        o3d.visualization.draw_geometries([pcd_rgbd])

    camera = CameraInfo(
        width, height, camera_instrinisc[0], camera_instrinisc[1], camera_instrinisc[2], camera_instrinisc[3], 1.0
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    depth_mask = (depth > 0)
    cloud_masked = cloud[depth_mask]
    # sample points random
    if len(cloud_masked) >= cfgs["num_point"]:
        idxs = np.random.choice(len(cloud_masked), cfgs["num_point"], replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs["num_point"] - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    if False:
        """ visualize sampled pointcloud """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32).reshape(-1, 3))
        o3d.visualization.draw_geometries([pcd])

    data_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs["voxel_size"],
                'feats': np.ones_like(cloud_sampled).astype(np.float32)}
    return data_dict

def data_process():
    root = cfgs["dataset_root"]
    camera_type = cfgs["camera"]

    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')))
    rgb = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')))
    seg = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')))
    meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat'))
    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    if False:
        """ visualize whole DEPTH pointcloud """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float32).reshape(-1, 3))
        o3d.visualization.draw_geometries([pcd])

    if False:
        """ visualize whole RGB-D pointcloud """
        height, width = depth.shape
        camera_instrinisc = np.array([intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]])
        depth_raw = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        color_raw = o3d.geometry.Image(np.ascontiguousarray(rgb).astype(np.uint8))
        rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 
                depth_scale=1e3, depth_trunc=10, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, *camera_instrinisc)
        pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_raw, intrinsic)
        # Twc = np.eye(4)
        # Twc[:3,:4] = meta['poses'][...,0]
        # Twc[3,3] = 1
        # # left multiply by Twc
        # pcd_rgbd.transform(np.linalg.inv(Twc))
        o3d.visualization.draw_geometries([pcd_rgbd])
    
    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)
    cloud_masked = cloud[mask]

    if False:
        """ visualize masked pointcloud """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32).reshape(-1, 3))
        mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0., 0., 0.])
        o3d.visualization.draw_geometries([pcd, mesh_frame_world])

    # sample points random
    if len(cloud_masked) >= cfgs["num_point"]:
        idxs = np.random.choice(len(cloud_masked), cfgs["num_point"], replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs["num_point"] - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs["voxel_size"],
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                }
    return ret_dict

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

class MyGraspNet():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.net = GraspNet(seed_feat_dim=cfgs["seed_feat_dim"], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(cfgs["checkpoint_path"])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs["checkpoint_path"], start_epoch))

        self.net.eval()
    def inference(self, pcs):

        data_dict = {'point_clouds': pcs.astype(np.float32),
                    'coors': pcs.astype(np.float32) / self.cfgs["voxel_size"],
                    'feats': np.ones_like(pcs).astype(np.float32)}
        batch_data = minkowski_collate_fn([data_dict])
        tic = time.time()
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        # Forward pass
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)
        # collision detection
        if self.cfgs["collision_thresh"] > 0:
            cloud = data_dict['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs["voxel_size_cd"])
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs["collision_thresh"])
            gg = gg[~collision_mask]

        # save grasps
        if self.cfgs["save_files"]:
            save_dir = os.path.join(self.cfgs["dump_dir"], scene_id, self.cfgs["camera"])
            save_path = os.path.join(save_dir, self.cfgs["index"] + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        toc = time.time()
        print('inference time: %fs' % (toc - tic))
        return gg

def inference(cfgs, data_input):
    batch_data = minkowski_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs["seed_feat_dim"], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs["checkpoint_path"])
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs["checkpoint_path"], start_epoch))

    net.eval()
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    # collision detection
    if cfgs["collision_thresh"] > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs["voxel_size_cd"])
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs["collision_thresh"])
        gg = gg[~collision_mask]

    # save grasps
    if cfgs["save_files"]:
        save_dir = os.path.join(cfgs["dump_dir"], scene_id, cfgs["camera"])
        save_path = os.path.join(save_dir, cfgs["index"] + '.npy')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        gg.save_npy(save_path)

    toc = time.time()
    print('inference time: %fs' % (toc - tic))
    return gg

def pointcloud_to_grasp(cfgs, pcs):
    data_dict = {'point_clouds': pcs.astype(np.float32),
                'coors': pcs.astype(np.float32) / cfgs["voxel_size"],
                'feats': np.ones_like(pcs).astype(np.float32)}
    gg = inference(cfgs, data_dict)
    return gg

def vis_grasp(pcs, gg):
    gg = gg.nms()
    gg = gg.sort_by_score()
    keep = 1
    if gg.__len__() > keep:
        gg = gg[:keep]
    grippers = gg.to_open3d_geometry_list()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
    o3d.visualization.draw_geometries([cloud, *grippers])      

if __name__ == '__main__':

    cfgs = parse_yaml("config.yaml")

    if not os.path.exists(cfgs["dump_dir"]):
        os.mkdir(cfgs["dump_dir"])

    scene_id = 'scene_' + cfgs["scene"]
    index = cfgs["index"]

    pcs = np.array(o3d.io.read_point_cloud("graspness_implementation/test_real/pcd_0.ply").points)
    gg = pointcloud_to_grasp(pcs)

    # data_dict = data_process_asgrasp()
    
    # # data_dict = data_process_gapartnet()
    # # data_dict = data_process()

    # if cfgs["infer"]:
    #     gg = inference(data_dict)

    if cfgs["vis"]:
        # pcs = data_dict['point_clouds']
        # gg = np.load(os.path.join(cfgs["dump_dir"], scene_id, cfgs["camera"], cfgs["index"] + '.npy'))
        # gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()
        keep = 10
        if gg.__len__() > keep:
            gg = gg[:keep]
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])      