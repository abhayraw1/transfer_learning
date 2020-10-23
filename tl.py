import torch


class TransferLearner:
    def __init__(self, model, optimizer, loss, scheduler, device, dataloader):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.dataloader = dataloader

    def train_on_batch(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def eval_on_batch(self, inputs, labels):
        self.model.eval()
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, labels)
        return loss

    def run_epoch(self, train=False):
        loss = 0.
        for inputs, labels in self.dataloader["train" if train else "val"]:
            if train:
                with torch.set_grad_enabled(True):
                    loss += self.train_on_batch(inputs, labels)
            else:
                with torch.set_grad_enabled(False):
                    loss += self.eval_on_batch(inputs, labels)
        return loss

    def learn(self, num_epochs):
        for i in range(num_epochs):
            print(f"Epoch {i+1}/{num_epochs}:")
            loss = self.run_epoch(train=True)
            print(f"\tTraining Loss: {loss}")
            loss = self.run_epoch(train=False)
            print(f"\tValidation Loss: {loss}\n")
