import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, num_users, num_items, learning_rate=0.0001,
                 num_epochs=100, lr_reduce_rate=0.9, lr_patience=5, stop_loss_patience=10):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate

        self.num_epochs = num_epochs

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.lr_patience = lr_patience  # Patience for learning rate reduction
        self.stop_patience = stop_loss_patience  # Patience for early stopping
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_reduce_rate, patience=self.lr_patience)
        # Weight decay is a regularization technique used to prevent overfitting by penalizing large weights
        # in the model's parameters. It works by adding a portion of the weights' magnitude to the loss function,
        # effectively encouraging the model to maintain smaller weight values. This is equivalent to
        # applying L2 regularization. In PyTorch, weight decay is specified as a parameter in the optimizer,
        # not within the model itself (e.g., the GMF class). For example, when initializing an optimizer like
        # Adam or SGD, you can set the weight_decay parameter:

        # A higher weight decay value increases the regularization strength, which can help reduce overfitting
        # but may also lead to underfitting if too strong.
        # e.g.: optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

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

        best_val_loss = float('inf')
        patience = self.stop_patience  # Number of epochs to wait for improvement before stopping
        wait = 0  # The counter for epochs waited without improvement

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

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                wait = 0
                # Save model state if this is the best performance so far
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                wait += 1
                if wait >= patience:
                    print(f'Stopping training. Best validation loss: {best_val_loss}')
                    break

            self.scheduler.step(validation_loss)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_dataloader)}, Validation Loss: {validation_loss}')


# Assuming model, train_dataloader, num_users, num_items are already defined
# trainer = Trainer(model, train_dataloader, num_users, num_items)
# trainer.train()
