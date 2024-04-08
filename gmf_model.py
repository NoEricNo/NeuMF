# gmf_model.py
import torch
import torch.nn as nn


class BiasedGMF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super(BiasedGMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)
        self.hidden_size = hidden_size

        # Embeddings for user and item biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Initialize biases to 0
        self.user_bias.weight.data.fill_(0.)
        self.item_bias.weight.data.fill_(0.)

        # Final prediction layer
        self.prediction = nn.Linear(hidden_size + 1, 1)  # Adjusted for bias term

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # Get bias embeddings for users and items
        # Avoid squeezing if the batch size is 1
        if user_ids.size(0) > 1:
            user_b = self.user_bias(user_ids).squeeze()
            item_b = self.item_bias(item_ids).squeeze()
        else:
            user_b = self.user_bias(user_ids).squeeze(1)
            item_b = self.item_bias(item_ids).squeeze(1)

        # Element-wise multiplication of user and item embeddings
        interaction = torch.mul(user_embed, item_embed)

        # Combine interaction and biases
        # Note: `torch.cat` is used to concatenate the bias terms along the feature dimension
        output = torch.cat([interaction, user_b.unsqueeze(1) + item_b.unsqueeze(1)], dim=1)

        prediction = self.prediction(output)
        return prediction.view(-1)

    # Optional: A prediction method similar to what you had for MLP
    def predict(self, user_id, item_id):
        with torch.no_grad():
            return self(user_id, item_id).squeeze()
