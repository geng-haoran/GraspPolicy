from algo.bc import BehaviorCloningModel, BehaviorCloningDataset, train

from sapien_gym import GraspEnv
import yaml
# from sapien_grasp_env import GraspEnv
from sapien_gym import GraspEnv
from utils.config_parser import parse_yaml
import os, glob, cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from utils.visu import save_imgs_to_video
from datetime import datetime
from PIL import Image
import argparse
import wandb
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

cfgs = parse_yaml("config.yaml")

env = GraspEnv(cfgs["sapien_env"])


def eval(model, try_num=3, device="cuda"):
    success_num = 0
    model.eval()
    with torch.no_grad():
        for try_i in range(try_num):
            for step in tqdm(list(range(500)), desc="eval"):
                # print(step)
                obs = env.reset(random = True)
                actions = model(torch.tensor(obs["state"], device=device), torch.tensor(obs["pc_xyz"], device=device))
                actions = {
                    "position": actions[:7].detach().cpu().numpy(),
                    "velocity": actions[7:14].detach().cpu().numpy(),
                    "gripper": actions[14:].detach().cpu().numpy()*10,
                }
                obs, reward, done, info = env.step(actions)
                if  env.objs[0].get_pose().p[2] >= 0.25:
                    print("success! step:", step, "success_num", success_num, "try_i", try_i)
                    success_num += 1
                    continue
            print("fail",try_i)
    if not try_num==0:
        success_rate = success_num/try_num
        print("success_rate", success_rate)
    else:
        success_rate = -1
    return success_rate

STATE_BASED = True
DATA_ROOT = "/raid/haoran/Project/GraspPolicy/demo_data/random_single_obj1101_2"
OBJ_IDS = ["005"]
BATCH_SIZE = 320
STATE_DIM = 35
ACTION_DIM = 16
PC_DIM = 3096
PC_FEA_DIM = 128
FEW_SHOT = False
DEVICE = "cuda"
current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M-%S")
LOG_DIR = f"logs/{formatted_time}"
os.makedirs(LOG_DIR, exist_ok=True)

wandb.init(
    project="GraspPolicy", 
    entity="haoran-geng",
    )

dataset = BehaviorCloningDataset(DATA_ROOT, OBJ_IDS, PC_DIM, DEVICE, FEW_SHOT)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("finish data loading")
model = BehaviorCloningModel(
    input_size=STATE_DIM, 
    output_size=ACTION_DIM, 
    hidden_layers=[128, 256, 256],
    pc_fea_dim=PC_FEA_DIM,
    state_based=STATE_BASED
)
model.to(DEVICE)
model = torch.load('/raid/haoran/Project/GraspPolicy/logs/11-02-22-08-11/model_10.pth')
# model.load_state_dict(torch.load('/raid/haoran/Project/GraspPolicy/logs/11-02-22-08-11/model_10.pth'))
print("finish model loading")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

num_epochs = 100
for epoch in range(num_epochs):
    
    success_rate = eval(model, try_num=10, device=DEVICE)
    train_loss = train(model, train_dataloader, optimizer, loss_fn)

    print(f'Epoch {epoch + 1}, Loss: {train_loss}, success_rate:{success_rate}')
    torch.save(model, f"{LOG_DIR}/model_{epoch}.pth")