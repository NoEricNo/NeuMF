# Import PyTorch and its neural network module, along with utilities for handling datasets and data loaders.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Define the Neural Collaborative Filtering (NCF) model class, extending PyTorch's Module class.
class NCF(nn.Module):
    # Initialize the NCF model with user/item counts, hidden layer size, and MLP structure base.
    def __init__(self, num_users, num_items, hidden_size, MLP_layers):
        super(NCF, self).__init__()  # Initialize the superclass (nn.Module) to set up the model.

        # Embedding layers for users and items, transforming IDs into dense vectors of size `hidden_size`.
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)

        # Dynamic MLP creation based on MLP_layers configuration
        self.MLP = self._create_mlp(hidden_size * 2, MLP_layers)

        # Prediction layer: the size of the last MLP layer to 1 output
        self.predict_layer = nn.Linear(MLP_layers[-1], 1)

    def _create_mlp(self, input_size, MLP_layers):
        layers = []
        for output_size in MLP_layers:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size  # Next layer's input size is the current layer's output size
        return nn.Sequential(*layers)

    # Forward pass: takes user and item IDs, and returns predictions.
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        # Concatenate user and item embeddings
        concat_embed = torch.cat([user_embed, item_embed], dim=1)  # Concatenate user/item embeddings.
        # Pass concatenated embeddings through MLP
        mlp_output = self.MLP(concat_embed)
        # Generate prediction

        preds = self.predict_layer(mlp_output)
        return preds.view(-1)

    # A prediction method that disables gradient calculations for inference.
    def predict(self, user_id, item_id):
        with torch.no_grad():  # Context manager that disables gradient computation.
            return self(user_id,
                        item_id).squeeze()  # Call the forward method and squeeze the result to remove extra dimensions.



