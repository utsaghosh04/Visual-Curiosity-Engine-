# Visual Curiosity Engine - Multi-Task Model

## Overview

This implementation provides a unified encoder-decoder model that simultaneously:
1. **Predicts curiosity heatmaps** - Pixel-wise regression to identify curious regions
2. **Generates questions** - Text generation for curiosity-driven questions

## Architecture

### Encoder
- **Backbone**: Pre-trained ResNet-34 (or ResNet-18/50)
- **Output**: Multi-scale feature maps with skip connections

### Heatmap Decoder (U-Net Style)
- **Architecture**: Fully convolutional decoder with skip connections
- **Output**: 1024×1024 heatmap (curiosity score per pixel)
- **Loss**: MSE or BCE loss

### Question Decoder
- **Architecture**: LSTM-based language model
- **Input**: Pooled image features from encoder
- **Output**: Question tokens (autoregressive generation)
- **Loss**: Cross-entropy loss

## Files Structure

```
Visual-Curiosity-Engine-/
├── data_loader.py          # Dataset and data loading utilities
├── model.py                # Model architecture definitions
├── trainer.py              # Training utilities and loss functions
├── model_idea_1.ipynb      # Main training notebook
└── requirements.txt        # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

Open `model_idea_1.ipynb` and run the cells sequentially:

1. **Setup**: Import modules and set device
2. **Data Loading**: Create train/val data loaders
3. **Vocabulary**: Build vocabulary from training questions
4. **Model Creation**: Initialize multi-task model
5. **Training**: Train the model with multi-task loss
6. **Evaluation**: Visualize results and test inference

### 3. Configuration

Key parameters in the notebook:

```python
BATCH_SIZE = 2              # Small for CPU/memory constraints
IMAGE_SIZE = 1024           # Input image size
NUM_EPOCHS = 50             # Training epochs
BACKBONE = 'resnet34'       # 'resnet18', 'resnet34', or 'resnet50'
```

## Training Strategy

### Multi-Task Loss
- **Heatmap Loss**: MSE between predicted and ground-truth heatmaps
- **Question Loss**: Cross-entropy over question tokens
- **Total Loss**: `L = λ₁ * L_heatmap + λ₂ * L_question`

### Transfer Learning
- Encoder initialized with ImageNet pre-trained weights
- Option to freeze encoder initially (uncomment in notebook)
- Fine-tune decoder heads on your data

### Data Augmentation
- Random horizontal flips
- Color jitter (brightness, contrast, saturation, hue)
- Applied during training only

## Data Format

### Annotations Structure

Each domain folder should contain:
- `annotations.json` with structure:
```json
{
  "annotations": [
    {
      "name": "img_001.png",
      "width": 1024,
      "height": 1024,
      "annotations": [
        {
          "xtl": 404.49,
          "ytl": 361.27,
          "xbr": 615.05,
          "ybr": 740.28,
          "attributes": {
            "curiosity_score": 3,
            "question_type": "why",
            "question": "why is the man sitting like that?"
          }
        }
      ]
    }
  ]
}
```

### Heatmap Generation

Ground-truth heatmaps are generated from bounding boxes using:
- Gaussian blur with configurable sigma (default: 20)
- Normalized curiosity scores (0-5 scale → 0-1)
- Multiple bboxes combined with max operation

## Model Usage

### Training

```python
from data_loader import create_data_loaders
from model import create_model
from trainer import Trainer, Vocabulary

# Create data loaders
train_loader, val_loader = create_data_loaders(...)

# Build vocabulary
vocab = Vocabulary()
# ... add sentences ...

# Create model
model = create_model(backbone='resnet34', vocab_size=len(vocab))

# Train
trainer = Trainer(model, train_loader, val_loader, vocab, device='cpu')
history = trainer.train(num_epochs=50)
```

### Inference

```python
# Load model
trainer.load_checkpoint('checkpoints/best_model.pth')

# Predict
model.eval()
with torch.no_grad():
    predictions = model(images, return_heatmap=True, return_question=True)
    
    heatmap = predictions['heatmap']
    question_logits = predictions['question']
    
    # Decode question
    tokens = question_logits.argmax(dim=-1)
    question = vocab.indices_to_sentence(tokens[0].tolist())
```

## Memory & CPU Optimization

For CPU-only training with 30GB RAM:

1. **Use ResNet-18** instead of ResNet-34 for faster training
2. **Small batch size** (1-2 images)
3. **Gradient accumulation** if needed (modify trainer)
4. **Freeze encoder** initially to reduce memory
5. **Reduce image size** if needed (e.g., 512×512 instead of 1024×1024)

## Checkpoints

Checkpoints are saved in `checkpoints/`:
- `best_model.pth` - Best model based on validation loss
- `checkpoint_epoch_N.pth` - Periodic checkpoints

Checkpoint includes:
- Model state
- Optimizer state
- Training history
- Vocabulary

## Tips

1. **Start with frozen encoder** - Unfreeze after decoder converges
2. **Monitor both losses** - Balance heatmap and question loss weights
3. **Use validation set** - Check for overfitting on small dataset
4. **Data augmentation** - Critical for 200-image dataset
5. **Learning rate** - Start with 1e-4, reduce if loss plateaus

## Troubleshooting

### Out of Memory
- Reduce batch size to 1
- Use ResNet-18 instead of ResNet-34
- Reduce image size
- Freeze encoder initially

### Slow Training
- Normal on CPU - expect hours for 50 epochs
- Consider training overnight
- Use smaller backbone if needed

### Poor Results
- Ensure data augmentation is enabled
- Check heatmap generation (visualize ground truth)
- Verify question tokenization
- Try different loss weights

## References

- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
- Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)
- Vision-Language Models (BLIP, ViT-GPT2, etc.)

