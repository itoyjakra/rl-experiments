from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pong_config as pc


class PongNetwork(nn.Module):
    """CNN network for Atari Pong"""

    def __init__(self, num_frames=4, num_actions=2):
        super().__init__()
        conv1 = nn.Conv2d(
            num_frames,
            pc.layer1.channels,
            kernel_size=pc.layer1.kernel_size,
            stride=pc.layer1.stride,
        )
        conv2 = nn.Conv2d(
            pc.layer1.channels,
            pc.layer2.channels,
            kernel_size=pc.layer2.kernel_size,
            stride=pc.layer2.stride,
        )
        conv3 = nn.Conv2d(
            pc.layer2.channels,
            pc.layer3.channels,
            kernel_size=pc.layer3.kernel_size,
            stride=pc.layer3.stride,
        )
        self.conv = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(pc.input_size_fc1, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, state):
        """Maps state to action."""
        # conv_out = self.conv(state)
        print(state.shape)
        print(self.conv(state).shape)
        conv_out = self.conv(state).view(state.size()[0], -1)
        print(conv_out.shape)
        return self.fc(conv_out)
