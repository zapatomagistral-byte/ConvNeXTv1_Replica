import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import warnings
import logging

# Silence annoying console warnings (especially useful when running on Colab)
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

# High-performance hardware optimization for NVIDIA GPUs
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 

# Device detection (leveraging GPU acceleration if available)
COLAB = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base directory for the dataset (Colab uses /content/data, local uses the project path)
root = '/content/data' if COLAB else './data'

# =====================================================
# 1. DATA AUGMENTATION & TRANSFORMS
# =====================================================

# Training transformations:
# - Random crop with padding of 4 to simulate spatial translations
# - Random horizontal flip
# - Standard normalization using official CIFAR-100 channel statistics
# - Random Erasing: Randomly masking patches to prevent the network from focusing on a single feature
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    ),
    transforms.RandomErasing(p=0.25)
])

# Validation and test transformations:
# - Only normalize the images to evaluate pure generalization performance
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

# =====================================================
# 2. DATASET INGESTION (CIFAR-100)
# =====================================================

train_data = datasets.CIFAR100(
    download=True,
    root=root,
    train=True,
    transform=train_transform
)

test_data = datasets.CIFAR100(
    download=True,
    root=root,
    train=False,
    transform=test_transform
)

# =====================================================
# 3. MODEL ARCHITECTURE CONFIGURATION
# =====================================================

# Output channel depth for each successive residual block
NeXTBlocks = [64, 64, 128, 128, 256, 256, 512, 512]

# Indices of the blocks where spatial downsampling (2x) is performed
Downsamplings = [2, 4, 6]

# Maximum drop probability for Stochastic Depth (Drop Path) in the final layers
Max_Prob = 0.3

# =====================================================
# 4. TRAINING HYPERPARAMETERS
# =====================================================

epochs = 300
batch_size = 128
Learning_Rate = 4e-3
warmup_epochs = 20

# =====================================================
# 5. ADVANCED DATA AUGMENTATION (Mixup & Cutmix)
# =====================================================

def mixup(x, y, alpha=0.8):
    """
    Linearly blends two random images in the batch and their corresponding labels
    based on a lambda ratio extracted from a Beta distribution.
    """
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

def cutmix(x, y, alpha=0.8):
    """
    Cuts a rectangular patch from one image and inserts it into another.
    The final labels are interpolated based on the exact pixel ratio of the replaced patch.
    """
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_rat = (1 - lam) ** 0.5
    cut_W = int(W * cut_rat)
    cut_H = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_W//2, 0)
    x2 = min(cx + cut_W//2, W)
    y1 = max(cy - cut_H//2, 0)
    y2 = min(cy + cut_H//2, H)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # Re-calculate precise lambda based on the exact pixel area replaced
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
    return x_mix, y, y[idx], lam

def aug_criterion(criterion, logits, y_a, y_b, lam):
    """
    Weighted loss function to support Mixup/Cutmix blending.
    Computes loss for both original labels and sums them weighted by lam.
    """
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# =====================================================
# 6. STOCHASTIC DEPTH (Random Layer DropPath)
# =====================================================

def drop(x, drop_prob, training):
    """
    Stochastic Depth (DropPath): Randomly disables the entire residual branch
    during training to act as a powerful regularizer against overfitting.
    """
    if drop_prob == 0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # Define shape for the random mask tensor (one per sample in the batch)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # Generate Bernoulli mask and scale the surviving branch to preserve expected activation magnitude
    random_tensor = keep_prob + torch.rand(shape, device=x.device)
    binary_tensor = torch.floor(random_tensor)
    return x / keep_prob * binary_tensor

def dropChances(numbloques):
    """
    Creates a linear decay schedule for Stochastic Depth probability.
    Earlier layers have a very low drop probability, while later, deeper layers
    have a higher probability (scaling linearly up to Max_Prob).
    """
    return [
        i / max(numbloques - 1, 1) * Max_Prob
        for i in range(numbloques)
    ]

# =====================================================
# 7. ARCHITECTURAL BLOCKS
# =====================================================

class Downsampling(nn.Module):
    """
    Spatial Reduction Layer: Halves the image dimensions (stride=2)
    and expands the output channel depth.
    """
    def __init__(self, In, Out):
        super().__init__()
        self.Downsample = nn.Sequential(
            nn.BatchNorm2d(In),
            nn.Conv2d(In, Out, stride=2, kernel_size=2)
        )
    def forward(self, x):
        return self.Downsample(x)

class NeXTBlock(nn.Module):
    """
    Modern Convolutional Block inspired by ConvNeXt:
    - Large 7x7 Depthwise Convolution (emulates the spatial mixing receptive field of a Transformer).
    - Batch Normalization layer.
    - 1x1 Convolution projecting channels to a higher dimension (Inverted Bottleneck, x4 channels).
    - GELU activation.
    - 1x1 Convolution projecting back to the original channel dimension.
    - Skip connection + Stochastic Depth (DropPath) regularization.
    """
    def __init__(self, in_channels, out_channels, drop_prob=0):
        super().__init__()
        self.inChannels = in_channels
        self.outChannels = out_channels
        self.stride = 1
        self.dropProb = drop_prob

        # Direct residual shortcut
        self.shortcut = nn.Identity()

        self.NeXTBlock = nn.Sequential(
            # Depthwise Convolution (groups=in_channels)
            nn.Conv2d(
                self.inChannels,
                self.inChannels,
                kernel_size=7,
                stride=self.stride,
                padding=3,
                bias=False,
                groups=self.inChannels
            ),
            nn.BatchNorm2d(self.inChannels),
            # Inverted Bottleneck expansion
            nn.Conv2d(
                self.inChannels,
                self.inChannels * 4,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.GELU(),
            # Channel projection projection back
            nn.Conv2d(
                self.inChannels * 4,
                self.outChannels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        funcion = self.NeXTBlock(x)
        
        # Apply DropPath to the residual function branch during training
        funcion = drop(
            funcion,
            drop_prob=self.dropProb,
            training=self.training
        )

        return funcion + shortcut

# =====================================================
# 8. COMPLETE NETWORK ARCHITECTURE
# =====================================================

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial stem projecting input RGB channels to initial channel depth (64)
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                NeXTBlocks[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(NeXTBlocks[0]),
            nn.GELU(),
        )

        self.In = NeXTBlocks[0]
        layers = []

        # Get the linear probability schedule for Stochastic Depth
        probs = dropChances(len(NeXTBlocks))

        # Sequential block construction
        for i, out_channels in enumerate(NeXTBlocks):
            # Interleave downsampling layers at specified stages
            if i in Downsamplings:
                layers.append(
                    Downsampling(self.In, out_channels)
                )
                self.In = out_channels
            
            # Append modern NeXTBlock
            layers.append(
                NeXTBlock(
                    self.In,
                    out_channels,
                    drop_prob=probs[i]
                )
            )
            self.In = out_channels

        self.Convs = nn.Sequential(*layers)

        # Global Average Pooling (GAP) collapsing spatial dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Linear Classification Head mapping final pooled channels to CIFAR-100 classes
        self.FC = nn.Linear(
            NeXTBlocks[-1],
            100
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.Convs(x)
        x = self.gap(x)
        
        # Flatten spatial dimensions
        x = x.view(x.shape[0], -1)
        
        # Protective 20% Dropout before classification
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.FC(x)
        return x

# =====================================================
# 9. INTEGRATED SCHEDULER BUILDER
# =====================================================

def scheduler_builder(optimizer):
    """
    Constructs a highly balanced dual learning rate scheduler:
    - Linear Warmup: Over the first 20 epochs, scales LR from 0.1% to 100% of base LR.
    - Cosine Annealing: Over the remaining 280 epochs, decays LR down to 1e-6 following a cosine curve.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

# =====================================================
# 10. MAIN HIGH-SPEED TRAINING LOOP
# =====================================================

if __name__ == '__main__':

    # High-speed parallel data loaders with system memory pinning and persistent workers
    train_dataLoader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2
    )

    test_dataLoader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Initialize model, cast to high-performance channels_last format, and compile graph
    model = ResNet().to(device)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)

    # Cross-entropy loss with label smoothing regularizer (0.1)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # AdamW optimizer with decoupled weight decay (L2 penalty)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Learning_Rate,
        weight_decay=0.05
    )

    scheduler = scheduler_builder(optimizer)

    # Gradient Scaler enabling stable mixed precision training (AMP)
    scaler = torch.amp.GradScaler('cuda')

    Print_Every = 20

    print("Starting training...")

    # Main epoch loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        batches = train_dataLoader

        # Inner batch iteration loop
        for images, labels in batches:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            
            # Perform Mixup or Cutmix data augmentation (50% probability each)
            r = np.random.rand()
            if r < 0.5:
                images, y_a, y_b, lam = mixup(images, labels)
            else:
                images, y_a, y_b, lam = cutmix(images, labels)
                
            optimizer.zero_grad(set_to_none=True)
            
            # Execute training step within mixed precision context
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = aug_criterion(criterion, logits, y_a, y_b, lam)

            # Compute backpropagation scaled gradients
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping to mitigate exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Record batch training statistics
            preds = logits.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            
        scheduler.step()

        # Regular evaluation on validation set
        if epoch % Print_Every == 0 or epoch == 1:
            model.eval()

            with torch.no_grad():
                test_correct = 0
                test_total = 0

                for images, labels in test_dataLoader:
                    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)

                    preds = outputs.argmax(dim=1)
                    test_total += labels.size(0)
                    test_correct += (preds == labels).sum().item()

                test_acc = (100 * test_correct / test_total)

            epoch_loss = running_loss / len(batches)
            train_acc = (100 * train_correct / train_total)
            elapsed_time = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Time: {elapsed_time:.1f}s"
            )
            # Periodic model checkpointing
            torch.save(model.state_dict(), "next100_full_weights.pth")
            print("Model saved.")
            
# Final weights checkpoint saving
torch.save(model.state_dict(), "next100_full_weights.pth")
print("Model saved.")