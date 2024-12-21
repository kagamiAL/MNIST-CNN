import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, out_ch: int = 16):
        super().__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(out_ch // 2 * 7 * 7, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, self.out_ch // 2 * 7 * 7)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
