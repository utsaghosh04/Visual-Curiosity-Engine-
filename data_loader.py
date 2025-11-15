"""
Data loader for Visual Curiosity Engine
Handles loading images, annotations, and creating heatmaps from bounding boxes
"""

import json
import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import cv2


def load_json_with_encoding(json_path):
    """
    Load JSON file trying multiple encodings to handle encoding issues.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        UnicodeDecodeError: If file cannot be decoded with any known encoding
    """
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        except Exception as e:
            # If it's a different error (like file not found), raise it
            if encoding == encodings[0]:  # Only raise on first attempt
                raise
    
    # If all encodings failed, raise an error
    raise ValueError(
        f"Could not decode {json_path} with any of the tried encodings: {encodings}"
    )


class CuriosityDataset(Dataset):
    """
    Dataset for loading images and annotations for curiosity heatmap and question generation.
    """
    
    def __init__(self, domain_folders=None, image_size=1024, augment=True, gaussian_sigma=20, samples=None, heatmap_dir='heatmap'):
        """
        Args:
            domain_folders: List of paths to domain folders (e.g., ['Domain_1_Images', 'Domain_2_Images'])
                           If None and samples is provided, samples will be used directly.
            image_size: Target image size (default 1024)
            augment: Whether to apply data augmentation
            gaussian_sigma: Sigma for Gaussian blur when creating heatmaps from bboxes (not used if heatmap_dir is provided)
            samples: Pre-loaded samples list (optional, for custom splits)
            heatmap_dir: Directory containing pre-generated heatmap images (default: 'heatmap')
        """
        self.image_size = image_size
        self.augment = augment
        self.gaussian_sigma = gaussian_sigma
        self.heatmap_dir = heatmap_dir
        
        # If samples are provided directly, use them
        if samples is not None:
            self.samples = samples
            print(f"Loaded {len(self.samples)} pre-selected samples")
            return
        
        # Otherwise, collect all images and annotations from domain folders
        self.samples = []
        for domain_folder in domain_folders:
            if not os.path.isdir(domain_folder):
                continue
            json_path = os.path.join(domain_folder, 'annotations.json')
            if not os.path.exists(json_path):
                continue
                
            annotations_data = load_json_with_encoding(json_path)
            # Handle both "annotations" and "images" keys (Domain_5 uses "images")
            items_list = annotations_data.get('annotations', []) or annotations_data.get('images', [])
            for item in items_list:
                img_name = item['name']
                img_path = os.path.join(domain_folder, img_name)
                
                if not os.path.exists(img_path):
                    continue
                
                # Get all annotations for this image
                bboxes = []
                questions = []
                question_types = []
                curiosity_scores = []
                
                for ann in item.get('annotations', []):
                    bbox = {
                        'xtl': ann['xtl'],
                        'ytl': ann['ytl'],
                        'xbr': ann['xbr'],
                        'ybr': ann['ybr']
                    }
                    bboxes.append(bbox)
                    questions.append(ann.get('attributes', {}).get('question', ''))
                    question_types.append(ann.get('attributes', {}).get('question_type', 'why'))
                    curiosity_scores.append(ann.get('attributes', {}).get('curiosity_score', 3))
                
                if bboxes:  # Only add if there are annotations
                    self.samples.append({
                        'image_path': img_path,
                        'bboxes': bboxes,
                        'questions': questions,
                        'question_types': question_types,
                        'curiosity_scores': curiosity_scores,
                        'width': item.get('width', image_size),
                        'height': item.get('height', image_size)
                    })
        
        print(f"Loaded {len(self.samples)} samples from {len(domain_folders)} domains")
    
    def __len__(self):
        return len(self.samples)
    
    def create_heatmap_from_bboxes(self, bboxes, scores, shape):
        """
        Create a Gaussian heatmap from bounding boxes.
        
        Args:
            bboxes: List of bbox dicts with 'xtl', 'ytl', 'xbr', 'ybr'
            scores: List of curiosity scores (0-5)
            shape: (height, width) of output heatmap
            
        Returns:
            numpy array of shape (height, width) with heatmap values
        """
        heatmap = np.zeros(shape, dtype=np.float32)
        
        for bbox, score in zip(bboxes, scores):
            # Normalize score to 0-1 range (assuming max score is 5)
            normalized_score = min(score / 5.0, 1.0)
            
            # Get bbox coordinates
            xtl = int(bbox['xtl'])
            ytl = int(bbox['ytl'])
            xbr = int(bbox['xbr'])
            ybr = int(bbox['ybr'])
            
            # Ensure coordinates are within bounds
            xtl = max(0, min(xtl, shape[1] - 1))
            ytl = max(0, min(ytl, shape[0] - 1))
            xbr = max(0, min(xbr, shape[1] - 1))
            ybr = max(0, min(ybr, shape[0] - 1))
            
            # Create a binary mask for this bbox
            mask = np.zeros(shape, dtype=np.float32)
            mask[ytl:ybr, xtl:xbr] = normalized_score
            
            # Apply Gaussian blur to the mask
            blurred = gaussian_filter(mask, sigma=self.gaussian_sigma)
            
            # Take maximum to combine multiple bboxes
            heatmap = np.maximum(heatmap, blurred)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Ensure contiguous array
        if not heatmap.flags['C_CONTIGUOUS']:
            heatmap = np.ascontiguousarray(heatmap)
        
        return heatmap
    
    def get_primary_question(self, questions, question_types):
        """
        Get the primary question (first non-empty question, or combine them)
        """
        if not questions:
            return "what is this?"
        
        # Return first non-empty question
        for q in questions:
            if q and q.strip():
                return q.strip()
        
        return "what is this?"
    
    def load_heatmap(self, image_path):
        """
        Load pre-generated heatmap from heatmap folder.
        Heatmap naming convention: img_001.png -> img_heatmap_001.png
        
        Args:
            image_path: Path to the original image
        
        Returns:
            numpy array of heatmap (grayscale, values 0-255 or 0-1)
        """
        # Get image filename
        img_name = os.path.basename(image_path)
        
        # Convert image filename to heatmap filename
        # img_001.png -> img_heatmap_001.png
        # Split the filename to extract base name and extension
        base_name, ext = os.path.splitext(img_name)
        
        # Insert '_heatmap' before the number (e.g., img_001 -> img_heatmap_001)
        # Find the last underscore followed by digits
        # Pattern to match: base_name_XXX.ext where XXX is digits
        match = re.match(r'(.+)_(\d+)$', base_name)
        if match:
            # Found pattern like img_001
            prefix = match.group(1)  # 'img'
            number = match.group(2)  # '001'
            heatmap_name = f"{prefix}_heatmap_{number}{ext}"  # 'img_heatmap_001.png'
        else:
            # Fallback: just add _heatmap before extension
            heatmap_name = f"{base_name}_heatmap{ext}"
        
        heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)
        
        if not os.path.exists(heatmap_path):
            # Fallback: try to create from bboxes if heatmap doesn't exist
            print(f"Warning: Heatmap not found at {heatmap_path}, using fallback")
            return None
        
        # Load heatmap image (assuming it's grayscale)
        heatmap = Image.open(heatmap_path).convert('L')
        heatmap = np.array(heatmap, dtype=np.float32)
        
        # Normalize to [0, 1] if values are in [0, 255]
        if heatmap.max() > 1.0:
            heatmap = heatmap / 255.0
        
        # Ensure contiguous array
        if not heatmap.flags['C_CONTIGUOUS']:
            heatmap = np.ascontiguousarray(heatmap)
        
        return heatmap
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Load pre-generated heatmap from heatmap folder
        heatmap = self.load_heatmap(sample['image_path'])
        
        # Fallback: create heatmap from bounding boxes if not found
        if heatmap is None:
            heatmap = self.create_heatmap_from_bboxes(
                sample['bboxes'],
                sample['curiosity_scores'],
                (sample['height'], sample['width'])
            )
        else:
            # Ensure heatmap matches image size
            if heatmap.shape != (sample['height'], sample['width']):
                # Resize heatmap to match original image size
                heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
                heatmap_pil = heatmap_pil.resize((sample['width'], sample['height']), Image.BILINEAR)
                heatmap = np.array(heatmap_pil, dtype=np.float32, copy=True, order='C') / 255.0
        
        # CRITICAL: Ensure heatmap is always contiguous from the start
        # This prevents any negative stride issues downstream
        heatmap = np.ascontiguousarray(heatmap)
        
        # Get primary question
        question = self.get_primary_question(sample['questions'], sample['question_types'])
        
        # Apply transforms
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = transforms.functional.hflip(image)
                # CRITICAL: After flip, ensure contiguous copy with C-order
                heatmap = np.fliplr(heatmap)
                heatmap = heatmap.copy(order='C')  # Force C-contiguous copy
            
            # Color jitter
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            if np.random.rand() > 0.5:
                image = color_jitter(image)
        
        # Resize to target size
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Resize heatmap to target size
        # CRITICAL: Always create a fresh contiguous copy before tensor conversion
        # This is the absolute fix for negative stride errors
        # We do this right before tensor conversion to ensure no negative strides
        # Method 1: Use copy with explicit C-order
        try:
            heatmap = heatmap.copy(order='C')
        except:
            # Fallback: Use np.array with explicit copy
            heatmap = np.array(heatmap, dtype=np.float32, copy=True, order='C')
        
        # Final safety check: ensure it's truly contiguous
        if not heatmap.flags['C_CONTIGUOUS'] or any(s < 0 for s in heatmap.strides):
            # Force a completely fresh array
            heatmap = np.ascontiguousarray(heatmap.copy())
        
        # Resize heatmap using interpolation
        # Convert to tensor - this should now be safe
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        heatmap_tensor = torch.nn.functional.interpolate(
            heatmap_tensor,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return {
            'image': image,
            'heatmap': heatmap_tensor,
            'question': question,
            'question_type': sample['question_types'][0] if sample['question_types'] else 'why',
            'image_path': sample['image_path']
        }


def create_data_loaders(base_dir, train_images_per_domain=28, val_images_per_domain=6,
                       batch_size=2, image_size=1024, num_workers=0, random_seed=42, heatmap_dir='heatmap'):
    """
    Create train and validation data loaders with random split per domain.
    
    Args:
        base_dir: Base directory containing domain folders
        train_images_per_domain: Number of images to randomly select from each domain for training (default: 28)
        val_images_per_domain: Number of images to randomly select from each domain for validation (default: 6)
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loader workers (0 for Windows compatibility)
        random_seed: Random seed for reproducibility
        heatmap_dir: Directory containing pre-generated heatmap images (default: 'heatmap')
    
    Returns:
        train_loader, val_loader
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # All domain folders
    all_domains = ['Domain_1_Images', 'Domain_2_Images', 'Domain_3_Images', 
                    'Domain_4_Images', 'Domain_5_Images']
    
    train_samples = []
    val_samples = []
    
    # Process each domain
    for domain_name in all_domains:
        domain_folder = os.path.join(base_dir, domain_name)
        
        if not os.path.isdir(domain_folder):
            print(f"Warning: Domain folder {domain_folder} not found, skipping...")
            continue
        
        json_path = os.path.join(domain_folder, 'annotations.json')
        if not os.path.exists(json_path):
            print(f"Warning: Annotations file {json_path} not found, skipping...")
            continue
        
        # Load all samples from this domain
        domain_samples = []
        try:
            annotations_data = load_json_with_encoding(json_path)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Error loading file: {json_path}")
            print(f"   Error: {e}")
            raise
        
        # Handle both "annotations" and "images" keys (Domain_5 uses "images")
        # Try "annotations" first, then "images" if "annotations" doesn't exist or is empty
        if 'annotations' in annotations_data and annotations_data['annotations']:
            items_list = annotations_data['annotations']
        elif 'images' in annotations_data and annotations_data['images']:
            items_list = annotations_data['images']
        else:
            items_list = []
        
        # Debug: Print if Domain_5 and items_list is empty
        if domain_name == 'Domain_5_Images' and len(items_list) == 0:
            print(f"DEBUG: Domain_5_Images - annotations key: {annotations_data.get('annotations', 'NOT_FOUND')}")
            print(f"DEBUG: Domain_5_Images - images key: {annotations_data.get('images', 'NOT_FOUND')}")
            print(f"DEBUG: Domain_5_Images - All keys: {list(annotations_data.keys())}")
        
        # Track statistics
        total_items = len(items_list)
        missing_images = 0
        items_without_bboxes = 0
        items_with_errors = 0
        
        for item in items_list:
            try:
                img_name = item.get('name')
                if not img_name:
                    items_with_errors += 1
                    continue
                    
                img_path = os.path.join(domain_folder, img_name)
                
                if not os.path.exists(img_path):
                    missing_images += 1
                    continue
                
                # Get all annotations for this image
                bboxes = []
                questions = []
                question_types = []
                curiosity_scores = []
                
                for ann in item.get('annotations', []):
                    try:
                        # Check if required bbox keys exist
                        if not all(key in ann for key in ['xtl', 'ytl', 'xbr', 'ybr']):
                            continue
                            
                        bbox = {
                            'xtl': ann['xtl'],
                            'ytl': ann['ytl'],
                            'xbr': ann['xbr'],
                            'ybr': ann['ybr']
                        }
                        bboxes.append(bbox)
                        questions.append(ann.get('attributes', {}).get('question', ''))
                        question_types.append(ann.get('attributes', {}).get('question_type', 'why'))
                        # Convert curiosity_score to int if it's a string
                        score = ann.get('attributes', {}).get('curiosity_score', 3)
                        if isinstance(score, str):
                            try:
                                score = int(score)
                            except (ValueError, TypeError):
                                score = 3
                        curiosity_scores.append(score)
                    except (KeyError, TypeError) as e:
                        # Skip this annotation if it's malformed
                        continue
                
                if bboxes:  # Only add if there are annotations
                    domain_samples.append({
                        'image_path': img_path,
                        'bboxes': bboxes,
                        'questions': questions,
                        'question_types': question_types,
                        'curiosity_scores': curiosity_scores,
                        'width': item.get('width', image_size),
                        'height': item.get('height', image_size)
                    })
                else:
                    items_without_bboxes += 1
            except Exception as e:
                items_with_errors += 1
                # Continue processing other items
                continue
        
        # Warn if no valid samples found
        if len(domain_samples) == 0:
            print(f"⚠️  Warning: Domain {domain_name} has no valid samples!")
            print(f"   Total items in JSON: {total_items}")
            print(f"   Missing images: {missing_images}")
            print(f"   Items without bboxes: {items_without_bboxes}")
            print(f"   Items with errors: {items_with_errors}")
            continue
        
        # Randomly shuffle domain samples
        np.random.shuffle(domain_samples)
        
        # Check if we have enough images
        total_available = len(domain_samples)
        required = train_images_per_domain + val_images_per_domain
        
        if total_available < required:
            print(f"Warning: Domain {domain_name} has only {total_available} images, "
                  f"but {required} are needed. Using all available images.")
            # Use what we have, prioritizing training set
            n_train = min(train_images_per_domain, total_available)
            n_val = min(val_images_per_domain, total_available - n_train)
        else:
            n_train = train_images_per_domain
            n_val = val_images_per_domain
        
        # Split into train and validation
        domain_train = domain_samples[:n_train]
        domain_val = domain_samples[n_train:n_train + n_val]
        
        train_samples.extend(domain_train)
        val_samples.extend(domain_val)
        
        print(f"Domain {domain_name}: {len(domain_train)} train, {len(domain_val)} validation "
              f"(total available: {total_available})")
    
    print(f"\nTotal: {len(train_samples)} training samples, {len(val_samples)} validation samples")
    
    # Create datasets with pre-selected samples
    # Use absolute path for heatmap_dir
    heatmap_path = os.path.join(base_dir, heatmap_dir) if not os.path.isabs(heatmap_dir) else heatmap_dir
    train_dataset = CuriosityDataset(samples=train_samples, image_size=image_size, augment=True, heatmap_dir=heatmap_path)
    val_dataset = CuriosityDataset(samples=val_samples, image_size=image_size, augment=False, heatmap_dir=heatmap_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Smaller batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader

