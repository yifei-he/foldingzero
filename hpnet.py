import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class hpnet(nn.Module):
    def __init__(self, l):
        super(hpnet, self).__init__()

        self.num_block = 5
        self.blocks = []
        for i in range(self.num_block):
            block = []
            if i == 0:
                block.append(nn.Conv2d(17, 128, kernel_size=3, stride=1, padding=1).to(device))
            else:
                block.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1).to(device))
            block.append(nn.BatchNorm2d(128).to(device))
            block.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1).to(device))
            block.append(nn.BatchNorm2d(128).to(device))
            self.blocks.append(block)
        self.policy_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1).to(device)
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1).to(device)

        self.out_size = l
        self.policy_fc = nn.Linear(l * l, 3).to(device)
        self.value_fc = nn.Linear(l * l, 1).to(device)
        # self.init_weights()

    def init_weights(self):
        for block in self.blocks:
            torch.nn.init.xavier_uniform_(block[0].weight)
            torch.nn.init.xavier_uniform_(block[2].weight)
        torch.nn.init.xavier_uniform_(self.policy_conv.weight)
        torch.nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, x):
        for i in range(self.num_block):
            temp = x
            x = self.blocks[i][0](x)
            x = self.blocks[i][1](x)
            x = F.relu(x)
            x = self.blocks[i][2](x)
            x = self.blocks[i][3](x)
            x = F.relu(x)
            if i != 0:
                x = x + temp
        p_out = F.relu(self.policy_conv(x))
        p_out = p_out.view(p_out.size(0), -1).squeeze(0)
        p_out = self.policy_fc(p_out)
        p_out = nn.Softmax(dim=0)(p_out)
        v_out = F.relu(self.value_conv(x))
        v_out = v_out.view(v_out.size(0), -1).squeeze(0)
        v_out = self.value_fc(v_out)
        return p_out, v_out
