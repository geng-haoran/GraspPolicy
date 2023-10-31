import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R

class GraspEnv():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)
        self.scene.add_ground(-0.8)
        physical_material = self.scene.create_physical_material(1, 1, 0.0)
        self.scene.default_physical_material = physical_material


        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        # if cfgs["use_viewer"]:
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

        # Robot
        # Load URDF
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(cfgs["robot"]["robot_urdf_path"])
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        # Set initial joint positions
        init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, \
                     0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        self.robot.set_qpos(init_qpos)

        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=100, damping=20)

        # table top
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
        builder.add_box_visual(half_size=[0.4, 0.4, 0.025])
        self.table = builder.build_kinematic(name='table')
        self.table.set_pose(sapien.Pose([0.56, 0, - 0.025]))

        # boxes
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.06], color=[1, 0, 0])
        self.red_cube = builder.build(name='red_cube')
        self.red_cube.set_pose(sapien.Pose([0.4, 0.3, 0.06]))

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.04])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.04], color=[0, 1, 0])
        self.green_cube = builder.build(name='green_cube')
        self.green_cube.set_pose(sapien.Pose([0.2, -0.3, 0.04]))

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.07])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.07], color=[0, 0, 1])
        self.blue_cube = builder.build(name='blue_cube')
        self.blue_cube.set_pose(sapien.Pose([0.6, 0.1, 0.07]))

        builder = self.scene.create_actor_builder()
        builder.add_collision_from_file(filename='assets/aligned/beer_can/visual_mesh.obj')
        builder.add_visual_from_file(filename='assets/aligned/beer_can/visual_mesh.obj')
        self.beer_can = builder.build(name='beer_can')
        self.beer_can.set_pose(sapien.Pose(p=[0.5, 0.1, 0.07]))

        self.setup_camera()
        self.setup_mplib_planner()
        
    
    def setup_mplib_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=self.cfgs["robot"]["robot_urdf_path"],
            srdf="assets/robot/panda/panda.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def setup_camera(self):
        near, far = 0.1, 100
        width, height = 640, 480
        self.camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        q = R.from_euler('xyz', [0, -40, 0], degrees=True).as_quat()
        self.camera.set_pose(sapien.Pose(p=[1.4, 0.00, 1], q=q))      

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.scene.step()
            if i % 4 == 0:
                # import pdb; pdb.set_trace()
                self.scene.update_render()
                self.viewer.render()

                import time
                s = time.time()
                self.camera.take_picture()
                t = time.time()
                print("pic", t - s)
                import time
                s = time.time()
                rgba = self.camera.get_float_texture('Color')  # [H, W, 4]
                t = time.time()
                print("rgba", t - s)
                position = self.camera.get_float_texture('Position')
                tt = time.time()
                print("position", tt - t)


                model_matrix = self.camera.get_model_matrix()
                points_opengl = position[..., :3][position[..., 3] < 1]
                points_color = rgba[position[..., 3] < 1][..., :3]
                points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

                # An alias is also provided
                # rgba = camera.get_color_rgba()  # [H, W, 4]
                rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                rgba_pil = Image.fromarray(rgba_img)
                rgba_pil.save('color.png')

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose_with_RRTConnect(self, pose):
        result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            print(result['status'])
            return -1
        self.follow_path(result)
        return 0

    def move_to_pose_with_screw(self, pose):
        result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
            if result['status'] != "Success":
                print(result['status'])
                return -1 
        
        print(result['position'].shape[0])
        self.follow_path(result)
        return 0
    
    def move_to_pose(self, pose, with_screw):
        if with_screw:
            return self.move_to_pose_with_screw(pose)
        else:
            return self.move_to_pose_with_RRT(pose)

    def demo(self, with_screw = True):
        poses = [[0.4, 0.3, 0.12, 0, 1, 0, 0],
                [0.2, -0.3, 0.08, 0, 1, 0, 0],
                [0.6, 0.1, 0.14, 0, 1, 0, 0]]
        for i in range(3):
            pose = poses[i]
            pose[2] += 0.2
            self.move_to_pose(pose, with_screw)
            self.open_gripper()
            pose[2] -= 0.12
            self.move_to_pose(pose, with_screw)
            self.close_gripper()
            pose[2] += 0.12
            self.move_to_pose(pose, with_screw)
            pose[0] += 0.1
            self.move_to_pose(pose, with_screw)
            pose[2] -= 0.12
            self.move_to_pose(pose, with_screw)
            self.open_gripper()
            pose[2] += 0.12
            self.move_to_pose(pose, with_screw)

if __name__ == '__main__':
    demo = GraspEnv(None)
    demo.demo()
