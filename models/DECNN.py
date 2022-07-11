import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, emb_dim, num_classes=2, dropout=0.5):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(emb_dim, 256)

        self.conv1 = nn.Conv1d(256, 128, 5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, 3, padding=1)

        self.dropout = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = nn.Conv1d(256, 256, 5, padding=2)

        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x_emb = self.linear1(x).unsqueeze(2)
        x_emb = self.dropout(x_emb)
        x_conv = nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = nn.functional.relu(self.conv5(x_conv))
        x_out = x_conv.reshape(x_conv.shape[0], -1)
        x = self.linear2(x_out)
        return x