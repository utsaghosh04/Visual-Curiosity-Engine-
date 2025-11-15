"""
Multi-task model for Visual Curiosity Engine
Combines heatmap prediction (U-Net style) with question generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms


class UNetDecoder(nn.Module):
    """
    U-Net style decoder for heatmap prediction.
    Uses skip connections from encoder layers.
    """
    
    def __init__(self, encoder_channels, decoder_channels=None, num_classes=1):
        """
        Args:
            encoder_channels: List of channel sizes from encoder (e.g., [64, 128, 256, 512])
            decoder_channels: List of channel sizes for decoder (default: reversed encoder_channels)
            num_classes: Number of output channels (1 for heatmap)
        """
        super(UNetDecoder, self).__init__()
        
        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels[:-1]))  # Skip the last one
        
        self.decoder_blocks = nn.ModuleList()
        
        # Build decoder blocks with skip connections
        in_channels = encoder_channels[-1]  # Start with the deepest encoder output
        
        for i, out_channels in enumerate(decoder_channels):
            # Get skip connection channels
            skip_channels = encoder_channels[-(i+2)]  # Corresponding encoder layer channels
            
            # Upsample block: transpose conv to upsample
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
            # After the conv_block, x will have out_channels channels
            # So the next transpose conv should take out_channels as input
            # (The conv_block takes out_channels + skip_channels as input and outputs out_channels)
            in_channels = out_channels
        
        # Conv blocks after skip connections
        self.conv_blocks = nn.ModuleList()
        for i, out_channels in enumerate(decoder_channels):
            skip_channels = encoder_channels[-(i+2)]
            in_conv_channels = out_channels + skip_channels
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_conv_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, encoder_features, skip_connections):
        """
        Args:
            encoder_features: Deepest encoder feature map (batch, channels, H, W)
            skip_connections: List of encoder feature maps for skip connections
        
        Returns:
            Heatmap prediction (batch, 1, H, W)
        """
        x = encoder_features
        
        # Reverse skip connections to match decoder order
        skip_connections = list(reversed(skip_connections))
        
        for i, (upsample_block, conv_block) in enumerate(zip(self.decoder_blocks, self.conv_blocks)):
            # Upsample
            x = upsample_block(x)
            
            # Concatenate with skip connection
            if i < len(skip_connections):
                skip = skip_connections[i]
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            # Apply conv block
            x = conv_block(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder with skip connections for U-Net decoder.
    """
    
    def __init__(self, backbone='resnet34', pretrained=True):
        super(ResNetEncoder, self).__init__()
        
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract layers for skip connections
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels (or 256 for resnet50)
        self.layer3 = resnet.layer3  # 256 channels (or 512 for resnet50)
        self.layer4 = resnet.layer4  # 512 channels (or 2048 for resnet50)
        
        # Get channel sizes
        if backbone == 'resnet50':
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            self.channels = [64, 64, 128, 256, 512]
    
    def forward(self, x):
        """
        Returns:
            encoder_output: Deepest feature map
            skip_connections: List of feature maps for skip connections
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip0 = x  # 64 channels, H/2, W/2
        
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        skip1 = x  # 64/256 channels, H/4, W/4
        
        x = self.layer2(x)
        skip2 = x  # 128/512 channels, H/8, W/8
        
        x = self.layer3(x)
        skip3 = x  # 256/1024 channels, H/16, W/16
        
        x = self.layer4(x)
        encoder_output = x  # 512/2048 channels, H/32, W/32
        
        skip_connections = [skip1, skip2, skip3]
        
        return encoder_output, skip_connections


class SimpleQuestionDecoder(nn.Module):
    """
    Simple LSTM-based question decoder.
    Alternative to Transformer-based decoder for CPU/memory efficiency.
    """
    
    def __init__(self, image_feature_dim=512, hidden_dim=256, vocab_size=10000, 
                 max_length=50, embedding_dim=256):
        super(SimpleQuestionDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Project image features to decoder hidden state
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, image_features, question_tokens=None, max_length=None):
        """
        Args:
            image_features: (batch, image_feature_dim) - pooled image features
            question_tokens: (batch, seq_len) - ground truth question tokens (for training)
            max_length: Maximum generation length (for inference)
        
        Returns:
            logits: (batch, seq_len, vocab_size) - predicted token logits
        """
        batch_size = image_features.size(0)
        max_len = max_length if max_length else self.max_length
        
        # Initialize hidden state from image features
        h0 = self.image_projection(image_features).unsqueeze(0)  # (1, batch, hidden_dim)
        c0 = torch.zeros_like(h0)
        
        if question_tokens is not None:
            # Training: teacher forcing
            embedded = self.embedding(question_tokens)  # (batch, seq_len, embedding_dim)
            lstm_out, _ = self.lstm(embedded, (h0, c0))
            logits = self.output_projection(lstm_out)
            return logits
        else:
            # Inference: autoregressive generation
            outputs = []
            input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=image_features.device)
            
            for _ in range(max_len):
                embedded = self.embedding(input_token)  # (batch, 1, embedding_dim)
                lstm_out, (h0, c0) = self.lstm(embedded, (h0, c0))
                logits = self.output_projection(lstm_out)  # (batch, 1, vocab_size)
                outputs.append(logits)
                
                # Get next token (greedy)
                input_token = logits.argmax(dim=-1)
            
            return torch.cat(outputs, dim=1)


class VisualCuriosityModel(nn.Module):
    """
    Multi-task model for curiosity heatmap prediction and question generation.
    """
    
    def __init__(self, backbone='resnet34', vocab_size=10000, max_question_length=50,
                 question_decoder_type='simple', use_question_head=True):
        """
        Args:
            backbone: ResNet backbone ('resnet18', 'resnet34', or 'resnet50')
            vocab_size: Vocabulary size for question generation
            max_question_length: Maximum question length
            question_decoder_type: 'simple' (LSTM) or 'transformer' (requires transformers lib)
            use_question_head: Whether to include question generation head
        """
        super(VisualCuriosityModel, self).__init__()
        
        self.use_question_head = use_question_head
        
        # Encoder
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=True)
        
        # Get encoder output channels
        encoder_channels = self.encoder.channels
        
        # Heatmap decoder (U-Net style)
        self.heatmap_decoder = UNetDecoder(
            encoder_channels=encoder_channels[1:],  # Skip the first 64-channel layer
            num_classes=1
        )
        
        # Question decoder
        if use_question_head:
            # Get image feature dimension (from deepest encoder layer)
            image_feature_dim = encoder_channels[-1]
            
            if question_decoder_type == 'simple':
                self.question_decoder = SimpleQuestionDecoder(
                    image_feature_dim=image_feature_dim,
                    hidden_dim=256,
                    vocab_size=vocab_size,
                    max_length=max_question_length,
                    embedding_dim=256
                )
            else:
                raise ValueError(f"Unsupported question_decoder_type: {question_decoder_type}")
            
            # Global average pooling for image features
            self.image_pooling = nn.AdaptiveAvgPool2d(1)
            self.image_flatten = nn.Flatten()
    
    def forward(self, x, question_tokens=None, return_heatmap=True, return_question=True):
        """
        Args:
            x: Input images (batch, 3, H, W)
            question_tokens: Ground truth question tokens (batch, seq_len) - for training
            return_heatmap: Whether to return heatmap prediction
            return_question: Whether to return question prediction
        
        Returns:
            Dictionary with 'heatmap' and/or 'question' predictions
        """
        outputs = {}
        
        # Encode image
        encoder_output, skip_connections = self.encoder(x)
        
        # Heatmap prediction
        if return_heatmap:
            heatmap = self.heatmap_decoder(encoder_output, skip_connections)
            # Upsample to original image size if needed
            if heatmap.shape[2:] != x.shape[2:]:
                heatmap = F.interpolate(
                    heatmap, size=x.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            outputs['heatmap'] = heatmap
        
        # Question generation
        if return_question and self.use_question_head:
            # Pool image features
            pooled = self.image_pooling(encoder_output)
            image_features = self.image_flatten(pooled)
            
            # Generate question
            question_logits = self.question_decoder(image_features, question_tokens)
            outputs['question'] = question_logits
        
        return outputs
    
    def freeze_encoder(self, freeze=True):
        """Freeze or unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
    
    def freeze_heatmap_decoder(self, freeze=True):
        """Freeze or unfreeze heatmap decoder weights."""
        for param in self.heatmap_decoder.parameters():
            param.requires_grad = not freeze


def create_model(backbone='resnet34', vocab_size=10000, use_question_head=True):
    """
    Factory function to create model.
    """
    model = VisualCuriosityModel(
        backbone=backbone,
        vocab_size=vocab_size,
        use_question_head=use_question_head
    )
    return model

