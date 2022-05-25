import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim / 4)
        self.fc2 = nn.Linear(input_dim / 8, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
