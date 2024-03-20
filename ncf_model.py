# Import PyTorch and its neural network module, along with utilities for handling datasets and data loaders.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Define the Neural Collaborative Filtering (NCF) model class, extending PyTorch's Module class.
class NCF(nn.Module):
    # Initialize the NCF model with user/item counts, hidden layer size, and MLP structure base.
    def __init__(self, num_users, num_items, hidden_size, MLP_struc_base):
        super(NCF, self).__init__()  # Initialize the superclass (nn.Module) to set up the model.
        self.num_users = num_users  # Store the number of users.
        self.num_items = num_items  # Store the number of items.
        self.hidden_size = hidden_size  # Store the size of the hidden layers.

        # Embedding layers for users and items, transforming IDs into dense vectors of size `hidden_size`.
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)

        # Multi-Layer Perceptron (MLP) for user embeddings, progressively decreasing in size.
        self.user_embedding_mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * MLP_struc_base), nn.ReLU(),  # First layer mapping to 4x base size, then ReLU.
            nn.Linear(4 * MLP_struc_base, 2 * MLP_struc_base), nn.ReLU(),
            # Second layer mapping to 2x base size, then ReLU.
            nn.Linear(2 * MLP_struc_base, MLP_struc_base), nn.ReLU()  # Final layer mapping to base size, then ReLU.
        )

        # Similarly, an MLP for item embeddings.
        self.item_embedding_mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * MLP_struc_base), nn.ReLU(),  # Matches the structure of the user MLP.
            nn.Linear(4 * MLP_struc_base, 2 * MLP_struc_base), nn.ReLU(),
            nn.Linear(2 * MLP_struc_base, MLP_struc_base), nn.ReLU()
        )

        # The prediction layer that takes concatenated user and item embeddings and outputs a single value.
        self.predict_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Combines user/item embeddings, then maps down.
            nn.ReLU(),  # Non-linearity.
            nn.Linear(hidden_size, 1),  # Final layer to reduce to a single prediction value.
        )

    # Forward pass: takes user and item IDs, and returns predictions.
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)  # Embed user IDs.
        user_embed = self.user_embedding_mlp(user_embed)  # Pass through MLP.
        item_embed = self.item_embedding(item_ids)  # Embed item IDs.
        item_embed = self.item_embedding_mlp(item_embed)  # Pass through MLP.
        concat_embed = torch.cat([user_embed, item_embed], dim=1)  # Concatenate user/item embeddings.
        preds = self.predict_layer(concat_embed)  # Get predictions from the final layer.
        return preds.view(-1)  # Flatten the predictions for output.

    # A prediction method that disables gradient calculations for inference.
    def predict(self, user_id, item_id):
        with torch.no_grad():  # Context manager that disables gradient computation.
            return self(user_id,
                        item_id).squeeze()  # Call the forward method and squeeze the result to remove extra dimensions.



