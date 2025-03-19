import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_Block_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_Block_A, self).__init__()
        
        self.branch1x1 = BasicConv2d(in_channels, 96, kernel_size=1)
        
        self.branch5x5_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(64, 96, kernel_size=5, padding=2)
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
        self.branch_pool = BasicConv2d(in_channels, 96, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Reduction_Block_A(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_Block_A, self).__init__()
        
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(192, 224, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(224, 256, kernel_size=3, stride=2)
    
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception_Block_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_Block_B, self).__init__()
        
        self.branch1x1 = BasicConv2d(in_channels, 384, kernel_size=1)
        
        self.branch7x7_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        
        self.branch7x7dbl_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(224, 224, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(224, 256, kernel_size=(1, 7), padding=(0, 3))
        
        self.branch_pool = BasicConv2d(in_channels, 128, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class Reduction_Block_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_Block_B, self).__init__()
        
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        
        self.branch7x7x3_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(320, 320, kernel_size=3, stride=2)
    
    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class Inception_Block_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_Block_C, self).__init__()
        
        self.branch1x1 = BasicConv2d(in_channels, 256, kernel_size=1)
        
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(384, 448, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_3 = BasicConv2d(448, 512, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_4a = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_4b = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch_pool = BasicConv2d(in_channels, 256, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3_a = self.branch3x3_2a(branch3x3)
        branch3x3_b = self.branch3x3_2b(branch3x3)
        branch3x3 = torch.cat([branch3x3_a, branch3x3_b], 1)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch3x3dbl_a = self.branch3x3dbl_4a(branch3x3dbl)
        branch3x3dbl_b = self.branch3x3dbl_4b(branch3x3dbl)
        branch3x3dbl = torch.cat([branch3x3dbl_a, branch3x3dbl_b], 1)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FaceNet(nn.Module):
    def __init__(self, num_classes=1000, embedding_size=128):
        super(FaceNet, self).__init__()
        
        # Input size: 299x299x3
        self.Conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)  # 149x149x32
        self.Conv2d_2a = BasicConv2d(32, 32, kernel_size=3)  # 147x147x32
        self.Conv2d_2b = BasicConv2d(32, 64, kernel_size=3, padding=1)  # 147x147x64
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # 73x73x64
        
        self.Conv2d_3b = BasicConv2d(64, 80, kernel_size=1)  # 73x73x80
        self.Conv2d_4a = BasicConv2d(80, 192, kernel_size=3)  # 71x71x192
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # 35x35x192
        
        # Inception blocks
        self.inception_a = nn.Sequential(
            Inception_Block_A(192),
            Inception_Block_A(384),
            Inception_Block_A(384)
        )  # 35x35x384
        
        self.reduction_a = Reduction_Block_A(384)  # 17x17x1024
        
        self.inception_b = nn.Sequential(
            Inception_Block_B(1024),
            Inception_Block_B(1024),
            Inception_Block_B(1024),
            Inception_Block_B(1024),
            Inception_Block_B(1024)
        )  # 17x17x1024
        
        self.reduction_b = Reduction_Block_B(1024)  # 8x8x1536
        
        self.inception_c = nn.Sequential(
            Inception_Block_C(1536),
            Inception_Block_C(1536)
        )  # 8x8x1536
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1x1536
        self.dropout = nn.Dropout(0.4)
        
        # Face embedding layer
        self.embedding = nn.Linear(1536, embedding_size)
        
        # L2 normalization layer for embedding vectors
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)
        
        # Classification layer (optional for training)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x, return_embedding=False):
        # Stem network
        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_2b(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b(x)
        x = self.Conv2d_4a(x)
        x = self.maxpool2(x)
        
        # Inception modules
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        
        # Global pooling and embedding
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Face embedding
        embedding = self.embedding(x)
        embedding = self.l2norm(embedding)
        
        if return_embedding:
            return embedding
        
        # Classification
        logits = self.classifier(embedding)
        return logits
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Loss function for face recognition
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

def create_model(pretrained=False, embedding_size=128, num_classes=None):
    model = FaceNet(num_classes=num_classes if num_classes else 1000, embedding_size=embedding_size)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available in this implementation")
    
    return model

def example_usage():
    model = create_model(embedding_size=128)
    
    x = torch.randn(1, 3, 299, 299)
    
    embedding = model(x, return_embedding=True)
    print(f"Embedding shape: {embedding.shape}")  # [1, 128]
    
    face1 = torch.randn(1, 3, 299, 299)
    face2 = torch.randn(1, 3, 299, 299)
    
    embedding1 = model(face1, return_embedding=True)
    embedding2 = model(face2, return_embedding=True)
    
    distance = torch.sum((embedding1 - embedding2) ** 2).item()
    print(f"Distance between faces: {distance}")
    
    threshold = 0.6
    same_person = distance < threshold
    print(f"Same person: {same_person}")

if __name__ == "__main__":
    # example_usage()
    model = create_model(embedding_size=128)

    summary(model, input_size=(3, 299, 299), batch_size=4, device="cpu")