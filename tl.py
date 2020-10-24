import copy
import torch


class TransferLearner:
    def __init__(self, model, optimizer, loss, scheduler, device, dataloader):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.dataloader = dataloader
        self.best_model_params = copy.deepcopy(self.model.state_dict())
        self.best_accuracy = 0.

    def train_on_batch(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, labels)
        accuracy = torch.sum(preds == labels.data)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

    def eval_on_batch(self, inputs, labels):
        self.model.eval()
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.loss(outputs, labels)
        accuracy = torch.sum(preds == labels.data)
        return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

    def run_epoch(self, train=False):
        loss = 0.
        accuracy = 0.
        num_samples = 0.
        for inputs, labels in self.dataloader["train" if train else "val"]:
            labels = labels.to(self.device)
            inputs = inputs.to(self.device)
            num_samples += labels.size(0)
            if train:
                with torch.set_grad_enabled(True):
                    l, a = self.train_on_batch(inputs, labels)
                    loss += l
                    accuracy += a
            else:
                with torch.set_grad_enabled(False):
                    l, a = self.eval_on_batch(inputs, labels)
                    loss += l
                    accuracy += a
        print(f"\t{accuracy}/{num_samples} correctly predicted!")
        if train:
            self.scheduler.step()
        else:
            if accuracy > self.best_accuracy:
                self.best_model_params = copy.deepcopy(self.model.state_dict())
                self.best_accuracy = accuracy
        return loss / num_samples, accuracy / num_samples

    def learn(self, num_epochs):
        stats = {'train': [], 'val': []}
        for i in range(num_epochs):
            print(f"Epoch {i + 1}/{num_epochs}:")
            loss, accuracy = self.run_epoch(train=True)
            stats['train'].append([loss, accuracy])
            print(f"\tTraining Loss: {loss}\n\tTraining Accuracy: {accuracy}")
            loss, accuracy = self.run_epoch(train=False)
            stats['val'].append([loss, accuracy])
            print(f"\tValidation Loss: {loss}\n\tValidation Accuracy: {accuracy}\n")
        return stats

    def save_model(self, name):
        torch.save(self.best_model_params, name)
