import torch
import torch.nn as nn
from mlp_model import MLP
from gmf_model import BiasedGMF


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mlp_layers, gmf_hidden_size, mlp_hidden_size, dropout_rate):
        super(NeuMF, self).__init__()

        # Initialize GMF and MLP models
        self.gmf = BiasedGMF(num_users, num_items, gmf_hidden_size)
        self.mlp = MLP(num_users, num_items, mlp_hidden_size, mlp_layers, dropout_rate, output_features=True)

        # The final prediction layer
        # The input size is the sum of GMF hidden size and the last MLP layer's size

        gmf_output_size = 1  # GMF model outputs a single score
        mlp_output_size = mlp_layers[-1]  # Size of MLP's final layer
        #print(f"gmf_output_size{gmf_output_size}, mlp_output_size{mlp_output_size}")
        self.final_prediction = nn.Linear(gmf_output_size + mlp_output_size, 1)

    def forward(self, user_ids, item_ids):
        # Get the outputs from GMF and MLP models
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)

        # Print shapes of the outputs for debugging
        #print(f"GMF output shape: {gmf_output.shape}")
        #print(f"MLP output shape: {mlp_output.shape}")

        # Add a dimension to make them [batch_size, 1] for concatenation
        gmf_output = gmf_output.unsqueeze(-1)  # Now torch.Size([16, 1])
        #mlp_output = mlp_output.unsqueeze(-1)  # Now torch.Size([16, 1])

        # Concatenate along dimension 1 to get torch.Size([16, 2])
        combined_output = torch.cat((gmf_output, mlp_output), dim=1)

        # Print the shape of the combined output before the final prediction layer
        #print(f"Combined output shape: {combined_output.shape}")

        # Pass the concatenated outputs through the final prediction layer
        prediction = self.final_prediction(combined_output)
        return prediction.view(-1)

    # Optional: A prediction method similar to what you had for MLP
    def predict(self, user_id, item_id):
        with torch.no_grad():
            return self(user_id, item_id).squeeze()
