# import torch
# import torch.nn as nn
# import numpy as np
from tqdm import tqdm

# import lightning.pytorch as lp
import torch, glob, os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import wandb
from algo.backbone import PointNetfeat
# from ..sapien_gym import GraspEnv
# from sapien_gym import GraspEnv

class BehaviorCloningDataset(Dataset):
    def __init__(self, data_root, obj_ids, pc_dim = 3096, device = "cuda", few_shot = False):
        self.data_root = data_root
        self.obj_ids = obj_ids
        self.pc_dim = pc_dim
        self.device = device
        data_paths = glob.glob(f"{self.data_root}/*/data/*.npy")
        self.valid_paths = []
        for path in data_paths:
            obj_id = path.split("/")[-3].split("_")[0]
            if obj_id in self.obj_ids:
                self.valid_paths.append(path)
        
        if few_shot:
            self.valid_paths = self.valid_paths[:32000]

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        data_path = self.valid_paths[idx]
        data = np.load(data_path, allow_pickle=True).item()
        state = data["obs"]["state"]
        pcs = data["obs"]["pc_xyz"]
        pcs_sampled = pcs[np.random.choice(pcs.shape[0], self.pc_dim, replace=False)]
        if "position" not in data["action"]:
            actions = np.zeros(16)
            actions[-2:] = np.array(data["action"]["gripper"])/10.0
        else:
            actions = np.concatenate((data["action"]["position"], data["action"]["velocity"], np.array(data["action"]["gripper"])/10.0), axis=0)

        return torch.tensor(state, dtype=torch.float32, device=self.device), \
                torch.tensor(pcs_sampled, dtype=torch.float32, device=self.device), \
                torch.tensor(actions, dtype=torch.float32, device=self.device)

class BehaviorCloningModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, pc_fea_dim=128, state_based=True):
        super(BehaviorCloningModel, self).__init__()
        self.state_based = state_based
        layers = []
        if not state_based:
            last_size = input_size + pc_fea_dim
        else:
            last_size = input_size
        for hidden_layer_size in hidden_layers:
            layers.append(torch.nn.Linear(last_size, hidden_layer_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_layer_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self.mlp = torch.nn.Sequential(*layers)
        if not state_based:
            self.pointnet = PointNetfeat(global_feat=True, feature_transform=False, glob_fea_dim=pc_fea_dim)

    def forward(self, state, pcs):
        if not self.state_based:
            pc_fea, _, _ = self.pointnet(pcs.transpose(1, 2))
            x = torch.cat((state, pc_fea), dim=1)
        else:
            x = state
        return self.mlp(x)
    
def train(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for state, pcs, actions in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(state, pcs)
        loss = loss_fn(outputs, actions)
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})
        # print(loss)
        total_loss += loss.item()
        # print(total_loss)
    average_loss = total_loss / len(dataloader)
    return average_loss

def eval(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    for state, pcs, actions in tqdm(dataloader, desc="Evaluating"):
        outputs = model(state, pcs)
        loss = loss_fn(outputs, actions)
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


if __name__ == '__main__':
    
    DATA_ROOT = "/raid/haoran/Project/GraspPolicy/demo_data/random_single_obj1101_2"
    OBJ_IDS = ["005"]
    BATCH_SIZE = 320
    STATE_DIM = 35
    ACTION_DIM = 16
    PC_DIM = 3096
    PC_FEA_DIM = 128
    DEVICE = "cuda"
    wandb.init(
        project="GraspPolicy", 
        entity="haoran-geng",
        )

    dataset = BehaviorCloningDataset(DATA_ROOT, OBJ_IDS, PC_DIM, DEVICE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("finish data loading")
    model = BehaviorCloningModel(
        input_size=STATE_DIM, 
        output_size=ACTION_DIM, 
        hidden_layers=[128, 256, 256],
        pc_fea_dim=PC_FEA_DIM
    )
    model.to(DEVICE)
    print("finish model loading")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, loss_fn)
        print(f'Epoch {epoch + 1}, Loss: {train_loss}')
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model = BehaviorCloningModel(input_size=10, output_size=5)
    # trainer = pl.Trainer(max_epochs=10)
    # trainer.fit(model, dataloader)

# class BehaviorCloning(object):
#     def __init__(self):
#         self.actor = None
        
#     ap = self.actor_critic.actor_mlp.parameters()
#     cp = self.actor_critic.critic_mlp.parameters()
#     self.actor_critic.backbone.unfreeze()
#     demo_states = self.demo_states.reshape(-1, *self.state_space.shape)
#     demo_pcs = self.demo_pcs.reshape(-1, *self.pc_space.shape)
#     demo_actions = self.demo_actions.reshape(-1, *self.action_space.shape)
#     demo_part_center = self.demo_part_center.reshape(-1, 3)
#     num_samples = demo_states.shape[0]

#     for ep in tqdm(range(self.model_cfg["bc_epochs"])):
#         l = int(num_samples / self.model_cfg["mb_size"])
#         for mb in range(l):
#             rand_idx = torch.from_numpy(np.random.choice(num_samples, size=self.model_cfg["mb_size"]))
#             obs = Observations(obs=demo_states[rand_idx], state=demo_states[rand_idx], points=demo_pcs[rand_idx])
#             obs_new = torch.cat((obs.obs[:, :22], obs.obs[:, 24:40], demo_part_center[rand_idx]), dim=1)
#             obs.obs = obs_new
#             act = demo_actions[rand_idx]
#             act_pred, _, _ = self.actor_critic.act_dagger(obs)
#             self.optimizer_bc.zero_grad()
#             loss = self.loss_function_bc(act, act_pred)
#             loss.backward()
#             self.optimizer_bc.step()
#             ###Log###
#             print("loss:", loss)
#             self.writer.add_scalar("BC/" + "loss", loss, ep * l + mb)
#             if self.wandb_writer is not None:
#                 self.wandb_writer.log({
#                     "BC/" + "loss", loss
#                 })