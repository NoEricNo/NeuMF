import torch
from ncf_FlexibleMLP import NCF
import ncf_dataset
from ncf_dataset import Dataset
from ncf_trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


dataset = Dataset(dataset_name="100k", batch_size=32)
dataset.chrono_split(train_ratio=0.6, val_ratio=0.2)
print(dataset.train_df.columns)
#dataset.reIndex_TrainValTestFiles()

train_dataloader = dataset.prepare_dataloader(dataset.train_df)
val_dataloader = dataset.prepare_dataloader(dataset.val_df)
num_users, num_items = ncf_dataset.get_user_item_counts(dataset.all_df)
if ncf_dataset.check_id_gaps(dataset.train_df):
    print("Warning: There are gaps in user IDs or item IDs.")

hidden_size = 16
#MLP_struc_base = 16
MLP_layers = [64, 32]
model = NCF(num_users, num_items, hidden_size, MLP_layers)

trainer = Trainer(model, train_dataloader, val_dataloader, num_users, num_items)
trainer.train()  # Start training
