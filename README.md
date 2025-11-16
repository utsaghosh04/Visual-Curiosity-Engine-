# Visual Curiosity Engine

A comprehensive project for predicting curiosity-inducing regions in images using deep learning and vision-language models. This repository contains multiple model architectures, training notebooks, and evaluation tools.

##  Project Structure

###  Notebooks

#### Core Model Training Notebooks

- **`proposed_model_architecture.ipynb`**
  - Multi-task model training notebook implementing a unified encoder-decoder architecture
  - Simultaneously predicts curiosity heatmaps (U-Net style decoder) and generates questions (LSTM-based decoder)
  - Uses pre-trained ResNet-34 encoder with transfer learning
  - Includes data loading, vocabulary building, model training, and evaluation

- **`existing_model_architecture.ipynb`**
  - Experimental notebook for saliency detection using OpenCV's fine-grained saliency detector
  - Demonstrates saliency map computation and visualization
  - Used for baseline comparisons and initial exploration

- **`blip_curiosity_net.ipynb`**
  - Fine-tuned BLIP-2 Vision-Language model for curiosity prediction
  - Architecture: Frozen BLIP-2 ViT backbone + domain context encoder + curiosity head
  - Predicts patch-level curiosity scores from 1024×1024 images
  - Includes masked supervision for sparse annotations

#### Vision-Language Model Notebooks

- **`Vision_Language.ipynb`**
  - Educational notebook explaining Vision-Language Models (VLMs), VQA, and VQG
  - Covers concepts like CLIP, BLIP, LLaVa, and their applications
  - Discusses integration strategies for curiosity prediction tasks

- **`vqa_model.ipynb`**
  - Extracts attention heatmaps from VQA (Visual Question Answering) models
  - Identifies question-relevant regions in images
  - Compares VQA attention with curiosity annotations
  - Generates hybrid heatmaps combining VQA attention with saliency

- **`vqa_fintune.ipynb`**
  - Fine-tunes a VQA-style BLIP model to output 14×14 curiosity heatmaps
  - Converts bounding-box annotations to soft heatmaps for supervision
  - Uses multi-loss function (MSE + Ranking + TV Smoothness)
  - Includes training, validation, and visualization sections

- **`vqg_finetune.ipynb`**
  - Fine-tunes a Visual Question Generation (VQG) model
  - Generates "WHY" questions similar to annotated curiosity regions
  - Uses cropped curiosity regions (bounding boxes) as input
  - Based on BLIP image-captioning model converted to question generation

#### Evaluation and Analysis Notebooks

- **`inference.ipynb`**
  - Model evaluation and metrics computation notebook
  - Evaluates BLIP-CuriosityNet and VQA-CuriosityNet models
  - Computes metrics: Pearson correlation, SSIM, MSE, Spearman rank correlation, NDCG@K
  - Generates visualization results for validation set

- **`heatmap_generation.ipynb`**
  - Generates hybrid curiosity heatmaps using Gaussian + Saliency methods
  - Creates "G-attended Saliency" heatmaps (full-image saliency amplified inside bounding boxes)
  - Generates heatmaps for 5 images from each domain
  - Uses spectral residual saliency and Gaussian filtering

###  Python Scripts

- **`model.py`**
  - Multi-task model architecture definitions
  - Contains UNetDecoder for heatmap prediction and QuestionDecoder for text generation
  - Implements encoder-decoder architecture with ResNet backbone

- **`trainer.py`**
  - Training utilities and loss functions
  - Includes Vocabulary class for question tokenization
  - Trainer class handles training loop, validation, checkpointing, and learning rate scheduling

- **`data_loader.py`**
  - Dataset and data loading utilities
  - CuriosityDataset class handles image loading, annotation parsing, and heatmap generation
  - Creates train/validation data loaders with data augmentation
  - Supports multiple encoding formats for JSON files

- **`build_vocab.py`**
  - Script to build vocabulary from training data
  - Extracts questions from dataset samples and creates word-to-index mappings
  - Used for question tokenization in multi-task model

- **`u2net.py`**
  - U2Net model implementation for saliency detection
  - Contains multi-scale loss functions (BCE, MSE, KL divergence)
  - Used as an alternative saliency detection method

###  Data Directories

- **`Domain_1_Images/` through `Domain_5_Images/`**
  - Contains image datasets from 5 different domains
  - Each domain folder includes:
    - 40 PNG images (1024×1024 resolution)
    - `annotations.json` file with bounding box annotations, questions, curiosity scores, and question types
    - Some domains include additional XML annotation files

- **`checkpoints/`**
  - Saved model checkpoints from training
  - Includes best model (`best_model.pth`) and epoch-specific checkpoints
  - Checkpoints contain model state, optimizer state, training history, and vocabulary

- **`curiosity_predictions_vqa/`**
  - Contains VQA model prediction results
  - Includes visualization images showing curiosity predictions
  - Contains `validation_metrics.json` with evaluation metrics

- **`heatmap/`**
  - Generated curiosity heatmap visualizations
  - Contains 200 PNG files showing heatmap overlays on images

- **`inference_results/`**
  - Inference visualization results
  - Contains 30 PNG files showing model predictions and comparisons

###  Documentation and Configuration

- **`README_MODEL.md`**
  - Detailed documentation for the multi-task model architecture
  - Includes architecture overview, training strategy, data format, usage examples, and troubleshooting

- **`requirements.txt`**
  - Python package dependencies
  - Includes PyTorch, torchvision, OpenCV, PIL, matplotlib, scipy, and other required libraries

###  Other Files

- **`Abstract (Visual Computing with AI_ML Project).pdf`**
  - Project abstract document

- **`Visual_Curiosity_Engine_Research_Paper.pdf`**
  - Research paper documentation

- **`annotations.xml`**
  - XML format annotations (alternative to JSON format)

- **`u2net.pth`**
  - Pre-trained U2Net model weights

- **`training_history.png`**
  - Visualization of training loss and metrics over epochs

- **`inference_sample.png`**
  - Sample inference visualization

##  Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Explore Notebooks**
   - Start with `proposed_model_architecture.ipynb` for the main multi-task model
   - Use `inference.ipynb` to evaluate pre trained BLIP and VQA models
   - Check `Vision_Language.ipynb` for background on VLMs

3. **Training**
   - Open `proposed_model_architecture.ipynb` and run cells sequentially
   - Configure batch size, image size, and number of epochs
   - Monitor training progress and validation metrics

4. **Evaluation**
   - Use `inference.ipynb` to load trained BLIP and VQA models and `proposed_model_architecture.ipynb` for proposed model and compute metrics
   - Visualize predictions and compare with ground truth annotations

##  Data Format

Annotations are stored in JSON format with the following structure:
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

##  Key Features

- **Multi-task Learning**: Simultaneous heatmap prediction and question generation
- **Multiple Architectures**: ResNet-based, BLIP-based, and VQA-based models
- **Transfer Learning**: Pre-trained vision encoders for better performance
- **Data Augmentation**: Heavy augmentation for small dataset (200 images)
- **Comprehensive Evaluation**: Multiple metrics including correlation, SSIM, and ranking metrics
- **Visualization Tools**: Heatmap generation and inference visualization

##  Model Architectures

1. **Multi-task ResNet Model**: ResNet-34 encoder with U-Net decoder for heatmaps and LSTM decoder for questions
2. **BLIP-CuriosityNet**: Fine-tuned BLIP-2 with domain context encoding
3. **VQA-CuriosityNet**: VQA-style model predicting curiosity heatmaps
4. **VQG Model**: Question generation from curiosity regions

##  Use Cases

- Identifying curiosity-inducing regions in images
- Generating questions about interesting image regions
- Visual attention prediction
- Educational content analysis
- Human curiosity modeling

##  References

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- BLIP: Bootstrapping Language-Image Pre-training
- Vision-Language Models (CLIP, BLIP-2, LLaVa)
- Visual Question Answering and Question Generation

