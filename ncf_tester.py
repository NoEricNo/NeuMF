import torch
from torch.utils.data import DataLoader
import numpy as np

class Evaluator:
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        mse_sum = 0  # Sum of mean squared errors
        mae_sum = 0  # Sum of mean absolute errors
        total_count = 0

        with torch.no_grad():  # No need to track gradients for evaluation
            for user_ids, item_ids, ratings in self.test_dataloader:

                predictions = self.model(user_ids, item_ids).view(-1)
                mse_sum += ((predictions - ratings) ** 2).sum().item()
                mae_sum += torch.abs(predictions - ratings).sum().item()
                total_count += ratings.size(0)

        mse = mse_sum / total_count
        mae = mae_sum / total_count
        rmse = np.sqrt(mse)

        return rmse, mae

# Example usage
# evaluator = Evaluator(neuMF_model, test_dataloader)
# rmse, mae = evaluator.evaluate()
# print(f"RMSE: {rmse}, MAE: {mae}")
