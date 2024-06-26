import ncf_dataset
from NeuMF import NeuMF
from ncf_dataset import Dataset
from ncf_trainer import Trainer
from datetime import datetime
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

# in the trainer class
# the patience is declared in the train function, remember that this value needs to be higher 
$ than the patience is the ReduceLROnPlateau scheduler
"""
max_epochs = 100
GMF_hidden_size = 2
MLP_hidden_size = 2
MLP_layers = [512, 256]
lr = 0.001  # learning rate.
batch_size = 512  # the powers of 2
lr_decay = 0.9
lr_patience = 10
stop_patience = lr_patience + 10
dropout_rate = 0.5
train_ratio = 0.5
val_ratio = 0.25
# something else?

# Load dataset and prepare dataloaders=========================================
dataset = Dataset(dataset_name="1m", batch_size=batch_size)
dataset.chrono_split(train_ratio=train_ratio, val_ratio=val_ratio)
print(dataset.train_df.columns)

train_dataloader = dataset.prepare_dataloader(dataset.train_df)
val_dataloader = dataset.prepare_dataloader(dataset.val_df)
test_dataloader = dataset.prepare_dataloader(dataset.test_df)

num_users, num_items = ncf_dataset.get_user_item_counts(dataset.all_df)
if ncf_dataset.check_id_gaps(dataset.train_df):
    print("Warning: There are gaps in user IDs or item IDs.")

# MLP_model = MLP(num_users, num_items, hidden_size, MLP_layers)
# GMF_model = BiasedGMF(num_users, num_items, hidden_size)

# Instantiate and train the model=============================================
NeuMF_model = NeuMF(num_users, num_items, MLP_layers, GMF_hidden_size, MLP_hidden_size, dropout_rate)
trainer = Trainer(NeuMF_model, train_dataloader, val_dataloader, num_users, num_items, lr,
                  max_epochs, lr_decay, lr_patience, stop_patience)

# Evaluate the model==========================================================
trainer.train()  # Start training
evaluator = Evaluator(NeuMF_model, test_dataloader)
rmse, mae = evaluator.evaluate()
print(f"RMSE: {rmse}, MAE: {mae}")
results = f"RMSE: {rmse}, MAE: {mae}"

# Define the output file name
# Generate a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"training_results_{timestamp}.txt"

# Write parameters and results to file
with open(output_file, "w") as file:
    file.write("Training Parameters:\n")
    file.write(f"Max Epochs: {max_epochs}\n")
    file.write(f"GMF Hidden Size: {GMF_hidden_size}\n")
    file.write(f"MLP Hidden Size: {MLP_hidden_size}\n")
    file.write(f"MLP Layers: {MLP_layers}\n")
    file.write(f"Learning Rate: {lr}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"Learning Rate Decay: {lr_decay}\n")
    file.write(f"Learning Rate Patience: {lr_patience}\n")
    file.write(f"Stop Patience: {stop_patience}\n")
    file.write(f"Dropout Rate: {dropout_rate}\n")
    file.write(f"Train Ratio: {train_ratio}\n")
    file.write(f"Validation Ratio: {val_ratio}\n\n")
    file.write("Results:\n")
    file.write(results + "\n")

print("Results and parameters have been written to", output_file)

# Adam: Combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that
# can handle sparse gradients on noisy problems. Adam is known for being effective in practice and requires little
# configuration – often the default settings work well.

# SGD (Stochastic Gradient Descent): Updates parameters in the opposite direction of the gradient.
# With momentum, SGD can avoid local minima and has a damping effect, leading to more stable convergence.
# Good for large datasets and settings where simplicity and transparency are preferred.

# RMSProp: Adapts the learning rate for each parameter, dividing the learning rate for a weight by a running average
# of the magnitudes of recent gradients for that weight. Especially effective in online and non-stationary settings.

# AdamW: A variant of Adam with a more effective way to handle weight decay. It decouples weight decay from the
# optimization steps, which can lead to better training stability and model performance.
