# NeXTNet - CIFAR100

If implementing ResNet from scratch was hard, at the time, this represented a whole new level of difficulty, and most importantly, a lot of learning.

This is the project that has meant the most to me so far and the one that has taught me the most. That's why I wanted to write down its story and insights myself.

---

## 1. The Architecture: Modernizing CNNs for the ViT Era

This, I believed, was most of what made **ConvNeXt V1** so important. The core idea behind it was to revive Convolutional Neural Networks (CNNs) to compete with Vision Transformers (ViTs) in vision tasks, simply by applying the design principles and lessons we learned from Transformers back onto CNNs.

### Key Architectural Shifts

#### 1. Transformer-Like Residual Blocks (NeXTBlock)
Instead of a traditional ResNet block—where you perform a full $3\times3$ convolution, then BatchNorm, then ReLU, followed by another $3\times3$ convolution, BatchNorm, and finally summing up with the original input before a final ReLU—ConvNeXt decides to make the block look as much like a Transformer block as possible:
* **Depthwise $7\times7$ Convolution First:** This attempts to imitate the way Transformers perform spatial mixing through self-attention. Making it depthwise (`groups=in_channels`) ensures it doesn't mess with the channel dimensions and remains highly efficient.
* **LayerNorm & Relu/GELU Substitution:** Standard ConvNeXt uses `LayerNorm` instead of `BatchNorm` (like a Transformer) and `GELU` instead of `ReLU`.
* **Inverted Bottleneck & Channel Expansion ($4\times$):** Inside the block, a $1\times1$ convolution expands the channel dimensions $4\times$, highly similar to the Feed-Forward Network (FFN) layer in Transformers. It then performs the `GELU` activation and projects down to the original channel dimension.
* **No Post-Summation Activation:** The input is summed directly with the output of the block (skip connection), but no activation function is applied after the summation. The signal enters the next block directly.

#### 2. Separate Downsampling Layers
Another change they did was to use downsampling layers completely separate from the main blocks, implementing them as a $2\times2$ convolution with stride 2.

#### 3. Patchifying Stem
The stem was changed to a large $4\times4$ convolution with stride 4, closely mimicking the aggressive patchifying of images performed in Vision Transformers (ViTs).

---

## 2. Surviving "Optimization Hell" & Hardware Realities

Ok, so is it a free upgrade just by changing the architecture? Well, **no**, and this is what made me learn the most so far.

When I first tested the architecture without the specific training recipe, I was shocked to see that:
1. It performed **worse** than a standard ResNet.
2. It trained **painfully slow** on the Google Colab T4 GPU.

This forced me to dive deep into hardware optimization. Despite depthwise convolutions being theoretically more efficient, standard CUDA kernels are notoriously unoptimized for them on GPU at small scales unless formatted properly. 

This made me learn how to optimize my code both using native PyTorch techniques and in the architecture itself:
* **The LayerNorm Bottleneck:** I initially tried to replicate `LayerNorm` applied to 2D channels by performing a `permute`, applying `LayerNorm`, and doing a `permute` back. This was incredibly slow! 
* **The Solution:** I pragmatically changed it to a regular `BatchNorm2d` because, for my specific use case, it resolved the entire training latency bottleneck without affecting performance.
* **Modern PyTorch Compilation:** I learned to leverage native PyTorch optimizations like `torch.compile(model)`, Automatic Mixed Precision (AMP) with `torch.amp.autocast('cuda')`, and forcing the `channels_last` memory format to unlock the full potential of NVIDIA Tensor Cores.

---

## 3. The Lesson: A "Better Architecture" Needs a Training Recipe

After making it through optimization hell, I could focus on the next big question: *Why didn't it just straight-up train better?*

As I was doing short tests, I was so confused by the "better architecture" not being just straight-up superior out of the box. This forced me to learn that deep learning is much more complex than just swapping layers. Another really important part of ConvNeXt V1 was its **training recipe**:
* **AdamW Optimizer:** Standardizing on AdamW with weight decay (`0.05`).
* **Linear Warmup + Cosine Decay:** Using a linear warmup of 20 epochs to stabilize gradients followed by a cosine annealing decay down to `1e-6`.
* **Aggressive Regularization & Data Augmentation:** Implementing absurdly powerful data augmentation techniques, specifically **Mixup** and **Cutmix** (at least for what I had seen until now), combined with **Stochastic Depth (DropPath)** and **Label Smoothing (0.1)**.

When I finally got the whole training recipe done and trained it for the full 300 epochs, the magic happened: **it outperformed ResNet by 5% test accuracy**, having a much higher margin between training and test accuracy (showing the power of the regularizers). 

---

## 4. Google Colab T4 GPU Training Results

The model was trained for **300 epochs** with manual Mixup/Cutmix blending, Stochastic Depth, and native graph compilation:

| Epoch | Learning Rate (LR) | Loss | Train Acc | Test Acc | Time/Epoch |
| :---: | :----------------: | :--: | :-------: | :------: | :--------: |
| **1** | 0.000204 | 4.5507 | 2.23% | 4.60% | 58.4s (Compilation) |
| **20** | 0.004000 | 3.3718 | 25.07% | 47.63% | 28.9s |
| **40** | 0.003950 | 3.0404 | 32.31% | 59.59% | 29.9s |
| **60** | 0.003802 | 2.9279 | 34.40% | 63.84% | 29.2s |
| **80** | 0.003564 | 2.8347 | 37.96% | 67.57% | 29.4s |
| **100** | 0.003247 | 2.7667 | 39.18% | 69.05% | 30.0s |
| **120** | 0.002868 | 2.7144 | 39.64% | 70.78% | 29.5s |
| **140** | 0.002445 | 2.6752 | 42.30% | 71.33% | 28.4s |
| **160** | 0.002000 | 2.5442 | 46.23% | 73.28% | 28.4s |
| **180** | 0.001556 | 2.4590 | 45.65% | 75.10% | 27.7s |
| **200** | 0.001133 | 2.3305 | 51.76% | 76.64% | 28.0s |
| **220** | 0.000754 | 2.2613 | 51.49% | 77.97% | 28.0s |
| **240** | 0.000437 | 2.1675 | 53.92% | 78.34% | 27.5s |
| **260** | 0.000199 | 2.0769 | 53.22% | 80.43% | 28.2s |
| **280** | 0.000051 | 1.9977 | 57.92% | 80.76% | 27.6s |
| **300** | 0.000001 | 2.0030 | 56.70% | **81.17%** | 28.5s |

*Note: Probably I could train it longer to achieve even more, but you can only do so much with a Colab free plan! I was extremely happy with the final **81.17%** result.*

---

## 5. Key Takeaways

Overall, I learned that:
1. The idea of a **"better architecture" is way more complex than it appears**.
2. It is **not only about changing the layers**, but also about the **training recipe** (optimizations, schedules, data augmentations).
3. **Hardware optimization matters:** How to properly structure code to run fast on GPU.

This was a really hard task, but totally worth it.