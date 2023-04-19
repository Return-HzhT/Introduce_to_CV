import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        # pass
        # ----------TODO------------
        # define a network
        # ----------TODO------------
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 72, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=72),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(nn.Linear(72 * 2 * 2, 48), nn.ReLU(),
                                nn.Linear(48, 10))

    def forward(self, x):
        # ----------TODO------------
        # network forwarding
        # ----------TODO------------
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    import torch
    from torch.utils.tensorboard import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir="\experiments/network_structure")
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=2)
    # Write a CNN graph.
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break
