"""
Training utilities for Visual Curiosity Engine
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os


class Vocabulary:
    """
    Simple vocabulary for question tokenization.
    """
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.idx = 4
        
    def add_sentence(self, sentence):
        """Add words from sentence to vocabulary."""
        words = sentence.lower().split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
    
    def sentence_to_indices(self, sentence, max_length=50):
        """Convert sentence to token indices."""
        words = sentence.lower().split()
        indices = [self.word2idx.get('<START>', 2)]
        
        for word in words[:max_length-2]:
            indices.append(self.word2idx.get(word, 1))  # 1 is <UNK>
        
        indices.append(self.word2idx.get('<END>', 3))
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(0)  # 0 is <PAD>
        
        return indices[:max_length]
    
    def indices_to_sentence(self, indices):
        """Convert token indices to sentence."""
        words = []
        for idx in indices:
            if idx == 0 or idx == 3:  # <PAD> or <END>
                break
            if idx == 2:  # <START>
                continue
            words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for heatmap and question generation.
    """
    
    def __init__(self, heatmap_weight=1.0, question_weight=1.0, heatmap_loss_type='mse'):
        """
        Args:
            heatmap_weight: Weight for heatmap loss
            question_weight: Weight for question loss
            heatmap_loss_type: 'mse' or 'bce'
        """
        super(MultiTaskLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.question_weight = question_weight
        
        if heatmap_loss_type == 'mse':
            self.heatmap_loss_fn = nn.MSELoss()
        elif heatmap_loss_type == 'bce':
            self.heatmap_loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported heatmap_loss_type: {heatmap_loss_type}")
        
        self.question_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD>
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with 'heatmap' and/or 'question' keys
            targets: Dict with 'heatmap' and/or 'question' keys
        
        Returns:
            total_loss, loss_dict
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Heatmap loss
        if 'heatmap' in predictions and 'heatmap' in targets:
            heatmap_loss = self.heatmap_loss_fn(predictions['heatmap'], targets['heatmap'])
            total_loss += self.heatmap_weight * heatmap_loss
            loss_dict['heatmap_loss'] = heatmap_loss.item()
        
        # Question loss
        if 'question' in predictions and 'question' in targets:
            # Reshape for cross-entropy: (batch * seq_len, vocab_size)
            question_pred = predictions['question'].reshape(-1, predictions['question'].size(-1))
            question_target = targets['question'].reshape(-1)
            question_loss = self.question_loss_fn(question_pred, question_target)
            total_loss += self.question_weight * question_loss
            loss_dict['question_loss'] = question_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


class Trainer:
    """
    Trainer class for multi-task model.
    """
    
    def __init__(self, model, train_loader, val_loader, vocab, device='cpu',
                 learning_rate=1e-4, heatmap_weight=1.0, question_weight=1.0,
                 save_dir='checkpoints'):
        """
        Args:
            model: VisualCuriosityModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab: Vocabulary instance
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate
            heatmap_weight: Weight for heatmap loss
            question_weight: Weight for question loss
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = MultiTaskLoss(
            heatmap_weight=heatmap_weight,
            question_weight=question_weight
        )
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        # Note: 'verbose' parameter was removed in newer PyTorch versions
        # Using without verbose for compatibility
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_heatmap_loss': [],
            'train_question_loss': [],
            'val_heatmap_loss': [],
            'val_question_loss': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_dict_avg = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            heatmaps = batch['heatmap'].unsqueeze(1).to(self.device)  # Add channel dim
            
            # Prepare question tokens
            questions = batch['question']
            question_tokens = []
            for q in questions:
                tokens = self.vocab.sentence_to_indices(q)
                question_tokens.append(tokens)
            question_tokens = torch.tensor(question_tokens, dtype=torch.long).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Create input tokens (shift by 1 for teacher forcing)
            input_tokens = question_tokens[:, :-1]
            target_tokens = question_tokens[:, 1:]
            
            predictions = self.model(
                images,
                question_tokens=input_tokens,
                return_heatmap=True,
                return_question=True
            )
            
            # Prepare targets
            targets = {
                'heatmap': heatmaps,
                'question': target_tokens
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_dict_avg:
                    loss_dict_avg[key] = 0.0
                loss_dict_avg[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hm_loss': f'{loss_dict.get("heatmap_loss", 0):.4f}',
                'q_loss': f'{loss_dict.get("question_loss", 0):.4f}'
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in loss_dict_avg:
            loss_dict_avg[key] /= num_batches
        
        return avg_loss, loss_dict_avg
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_dict_avg = {}
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # Move to device
            images = batch['image'].to(self.device)
            heatmaps = batch['heatmap'].unsqueeze(1).to(self.device)
            
            # Prepare question tokens
            questions = batch['question']
            question_tokens = []
            for q in questions:
                tokens = self.vocab.sentence_to_indices(q)
                question_tokens.append(tokens)
            question_tokens = torch.tensor(question_tokens, dtype=torch.long).to(self.device)
            
            # Forward pass
            input_tokens = question_tokens[:, :-1]
            target_tokens = question_tokens[:, 1:]
            
            predictions = self.model(
                images,
                question_tokens=input_tokens,
                return_heatmap=True,
                return_question=True
            )
            
            # Prepare targets
            targets = {
                'heatmap': heatmaps,
                'question': target_tokens
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Update statistics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_dict_avg:
                    loss_dict_avg[key] = 0.0
                loss_dict_avg[key] += value
        
        # Average losses
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        for key in loss_dict_avg:
            loss_dict_avg[key] /= num_batches
        
        return avg_loss, loss_dict_avg
    
    def train(self, num_epochs, save_every=5):
        """Train the model for multiple epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_loss_dict = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_loss_dict = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_heatmap_loss'].append(train_loss_dict.get('heatmap_loss', 0))
            self.history['train_question_loss'].append(train_loss_dict.get('question_loss', 0))
            self.history['val_heatmap_loss'].append(val_loss_dict.get('heatmap_loss', 0))
            self.history['val_question_loss'].append(val_loss_dict.get('question_loss', 0))
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train HM Loss: {train_loss_dict.get('heatmap_loss', 0):.4f} | "
                  f"Train Q Loss: {train_loss_dict.get('question_loss', 0):.4f}")
            print(f"Val HM Loss: {val_loss_dict.get('heatmap_loss', 0):.4f} | "
                  f"Val Q Loss: {val_loss_dict.get('question_loss', 0):.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'vocab': self.vocab
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.vocab = checkpoint.get('vocab', self.vocab)
        print(f"Loaded checkpoint from {path}")

