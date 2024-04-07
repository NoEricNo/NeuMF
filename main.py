import torch
from mlp_model import MLP
from gmf_model import BiasedGMF
from NeuMF import NeuMF
import ncf_dataset
from ncf_dataset import Dataset
from ncf_trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from ncf_tester import Evaluator
"""
# for GMF
# need to tune learning rate in trainer (class)
# need to tune hidden size in GMF (class)
# need to tune batch_size in dataset (class)
# need to tune weight_decay in trainer (class)

# for MLP
# need to tune learning rate in trainer (class)
# need to tune hidden size in MLP (class)
# need to tune batch_size in dataset (class)
# need to tune weight_decay in trainer (class)
# need to tune MLP struct in MLP (class)
# need to tone dropout rate in MLP (class)

# for both
# remember to make weight decay value higher for GMF since GMF does not have dropout mechanism
# maybe we should consider different optimizer aside from ADAM
"""

dataset = Dataset(dataset_name="100k", batch_size=8)
dataset.chrono_split(train_ratio=0.6, val_ratio=0.2)
print(dataset.train_df.columns)

train_dataloader = dataset.prepare_dataloader(dataset.train_df)
val_dataloader = dataset.prepare_dataloader(dataset.val_df)
test_dataloader = dataset.prepare_dataloader(dataset.test_df)

num_users, num_items = ncf_dataset.get_user_item_counts(dataset.all_df)
if ncf_dataset.check_id_gaps(dataset.train_df):
    print("Warning: There are gaps in user IDs or item IDs.")

GMF_hidden_size = 16
MLP_hidden_size = 16
MLP_layers = [64, 32, 16]
# MLP_model = MLP(num_users, num_items, hidden_size, MLP_layers)
# GMF_model = BiasedGMF(num_users, num_items, hidden_size)
NeuMF_model = NeuMF(num_users, num_items, MLP_layers, GMF_hidden_size, MLP_hidden_size )
trainer = Trainer(NeuMF_model, train_dataloader, val_dataloader, num_users, num_items)
trainer.train()  # Start training
evaluator = Evaluator(NeuMF_model, test_dataloader)
rmse, mae = evaluator.evaluate()
print(f"RMSE: {rmse}, MAE: {mae}")