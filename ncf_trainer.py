import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, num_users, num_items, learning_rate=0.001,
                 batch_size=512, num_epochs=100):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)

    def validate(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        count = 0

        with torch.no_grad():  # No gradients needed
            for user_ids, item_ids, ratings in self.val_dataloader:
                # Move data to the appropriate device (e.g., GPU)
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
                count += 1

        average_loss = total_loss / count
        return average_loss

    def train(self):

        self.model.train()  # Set the model to training mode
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            # Update learning rate scheduler

            for user_ids, item_ids, ratings in self.train_dataloader:
                # Optionally move data to GPU
                # user_ids, item_ids, ratings = user_ids.cuda(), item_ids.cuda(), ratings.cuda()

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            validation_loss = self.validate()
            print(f'Epoch: {epoch}, Validation Loss: {validation_loss}')
            self.scheduler.step(validation_loss)

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_dataloader)}')


# Assuming model, train_dataloader, num_users, num_items are already defined
#trainer = Trainer(model, train_dataloader, num_users, num_items)
#trainer.train()
