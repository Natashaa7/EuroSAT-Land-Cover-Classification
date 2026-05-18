import torch
import torch.nn as nn
from torchvision import models

# Baseline Model: Simple CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),              # Activation function (introduces non-linearity)
            nn.MaxPool2d(2),        # Downsamples image by factor of 2 (reduces H and W)
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)         # Again reduces spatial size by factor of 2 (kernel, stride)
        )

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Flatten(),  # Converts feature maps into 1D vector
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),

            # Output layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Pass input through convolutional layers
        x = self.conv(x)

        # Flatten and pass through fully connected layers
        x = self.fc(x)

        # Output logits (no softmax here; loss function will handle it)
        return x
    
# Improved Model: Deeper CNN with BatchNorm and Dropout
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # NEW Block 3 (added depth)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 64 channels, size becomes 8×8 after 3 pools
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),

            nn.Dropout(0.5),   # Prevent overfitting

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    
# ResNet Model: Pretrained ResNet18
def get_resnet(num_classes):
    model = models.resnet18(pretrained=True)

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model