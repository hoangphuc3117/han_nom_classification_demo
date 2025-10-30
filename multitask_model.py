"""
Multi-Task Model Architecture without Streamlit dependencies
Pure PyTorch implementation for testing and inference

This module contains the Multi-Task ResNet50 model with CBAM for:
- Hierarchical document classification (3 levels)
- Rotation angle prediction (4 angles: 0°, 90°, 180°, 270°)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==================== CONSTANTS ====================
MAIN_CATEGORIES = {"SinoNom": 0, "NonSinoNom": 1}
DOC_TYPES = {"Thong_thuong": 0, "Hanh_chinh": 1, "Ngoai_canh": 2}
TEXT_DIRECTIONS = {"Doc": 0, "Ngang": 1}
ROTATION_ANGLES = {0: 0, 90: 1, 180: 2, 270: 3}

# Reverse mappings for inference
INV_MAIN_CATEGORIES = {v: k for k, v in MAIN_CATEGORIES.items()}
INV_DOC_TYPES = {v: k for k, v in DOC_TYPES.items()}
INV_TEXT_DIRECTIONS = {v: k for k, v in TEXT_DIRECTIONS.items()}
INV_ROTATION_ANGLES = {v: k for k, v in ROTATION_ANGLES.items()}

IMAGE_SIZE = (224, 224)
IMAGE_DEPTH = 3

# ==================== MODEL ARCHITECTURE ====================

class ChannelAttention(nn.Module):
    """Channel Attention trong CBAM."""
    
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention trong CBAM."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(gate_channels, reduction_ratio)
        self.spatial = spatial
        if self.spatial:
            self.spatial_att = SpatialAttention()

    def forward(self, x):
        x_out = x * self.channel_att(x)
        if self.spatial:
            x_out = x_out * self.spatial_att(x_out)
        return x_out

class BottleneckCBAM(nn.Module):
    """Bottleneck block với CBAM attention."""
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = use_cbam
        self.cbam = CBAM(planes * self.expansion) if use_cbam else None
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.cbam:
            out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class MultiTaskResNet50(nn.Module):
    """
    ResNet50 với CBAM cho multi-task learning:
    - Hierarchical classification (3 levels)
    - Rotation classification (4 angles: 0°, 90°, 180°, 270°)
    """
    
    def __init__(self, use_cbam=True, image_depth=3, num_classes=[2,3,2], num_rotations=4):
        super().__init__()
        # Định nghĩa ResNet backbone
        self.inplanes = 64
        self.conv1 = nn.Conv2d(image_depth, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BottleneckCBAM, 64, 3, use_cbam=use_cbam)
        self.layer2 = self._make_layer(BottleneckCBAM, 128, 4, stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(BottleneckCBAM, 256, 6, stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(BottleneckCBAM, 512, 3, stride=2, use_cbam=use_cbam)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Hierarchical classification heads
        self.fc_main = nn.Linear(2048, num_classes[0])
        self.fc_doc = nn.Linear(2048, num_classes[1])
        self.fc_text = nn.Linear(2048, num_classes[2])
        
        # Rotation classification head
        self.fc_rotation = nn.Linear(2048, num_rotations)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1, use_cbam=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=use_cbam))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Hierarchical classification outputs
        out_main = self.fc_main(x)
        out_doc = self.fc_doc(x)
        out_text = self.fc_text(x)
        
        # Rotation classification output
        out_rotation = self.fc_rotation(x)
        
        # Trả về cả outputs cho hierarchical classification và rotation
        return [out_main, out_doc, out_text, out_rotation]

# ==================== UTILITY FUNCTIONS ====================

def load_resnet50_weights_to_custom(model, imagenet_weights=None):
    """
    Load pretrained weights từ torchvision ResNet50 vào custom ResNet50 + CBAM.
    
    Args:
        model: Instance của MultiTaskResNet50
        imagenet_weights: Path tới pretrained weights (optional)
    """
    if imagenet_weights is None:
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet50 = models.resnet50()
        resnet50.load_state_dict(torch.load(imagenet_weights))
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in resnet50.state_dict().items() 
                      if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded pretrained weights for Multi-Task ResNet50 backbone!")

def preprocess_image(image, image_size=IMAGE_SIZE):
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image object
        image_size: Tuple of (height, width)
    
    Returns:
        Preprocessed image tensor
    """
    try:
        # Convert to RGB if needed
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image_tensor, device='cpu'):
    """
    Make prediction on preprocessed image with multi-task model.
    
    Args:
        model: Trained MultiTaskResNet50 model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing predictions and confidences
    """
    try:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            predictions = model(image_tensor)
            
            # Convert to probabilities
            prob_1 = F.softmax(predictions[0], dim=1)
            prob_2 = F.softmax(predictions[1], dim=1)
            prob_3 = F.softmax(predictions[2], dim=1)
            prob_rotation = F.softmax(predictions[3], dim=1)
            
            # Get predicted classes
            pred_1 = prob_1.argmax(dim=1).item()
            pred_2 = prob_2.argmax(dim=1).item()
            pred_3 = prob_3.argmax(dim=1).item()
            pred_rotation = prob_rotation.argmax(dim=1).item()
            
            # Get confidence scores
            conf_1 = prob_1.max().item()
            conf_2 = prob_2.max().item()
            conf_3 = prob_3.max().item()
            conf_rotation = prob_rotation.max().item()
            
            # Apply hierarchical logic
            main_category = INV_MAIN_CATEGORIES[pred_1]
            
            if main_category == "SinoNom":
                doc_type = INV_DOC_TYPES[pred_2]
                if doc_type == "Thong_thuong":
                    text_direction = INV_TEXT_DIRECTIONS[pred_3]
                else:
                    text_direction = "N/A"
                    conf_3 = 0.0
            else:
                doc_type = "N/A"
                text_direction = "N/A"
                conf_2 = 0.0
                conf_3 = 0.0
            
            # Get rotation angle
            rotation_angle = INV_ROTATION_ANGLES[pred_rotation]
            
            return {
                'main_category': main_category,
                'main_category_confidence': conf_1,
                'document_type': doc_type,
                'document_type_confidence': conf_2,
                'text_direction': text_direction,
                'text_direction_confidence': conf_3,
                'rotation_angle': rotation_angle,
                'rotation_confidence': conf_rotation,
                'raw_probabilities': {
                    'level_1': prob_1.cpu().squeeze().tolist(),
                    'level_2': prob_2.cpu().squeeze().tolist(),
                    'level_3': prob_3.cpu().squeeze().tolist(),
                    'rotation': prob_rotation.cpu().squeeze().tolist()
                }
            }
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return None

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Create and initialize model
    print("Creating Multi-Task ResNet50 model...")
    model = MultiTaskResNet50(
        use_cbam=True,
        image_depth=IMAGE_DEPTH,
        num_classes=[len(MAIN_CATEGORIES), len(DOC_TYPES), len(TEXT_DIRECTIONS)],
        num_rotations=len(ROTATION_ANGLES)
    )
    
    # Load pretrained backbone weights
    print("Loading pretrained ResNet50 weights...")
    load_resnet50_weights_to_custom(model)
    
    print(f"✅ Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Example inference
    print("\nExample inference with random input:")
    dummy_input = torch.randn(1, IMAGE_DEPTH, IMAGE_SIZE[0], IMAGE_SIZE[1])
    with torch.no_grad():
        outputs = model(dummy_input)
        print(f"Output shapes:")
        print(f"  - Main category: {outputs[0].shape}")
        print(f"  - Document type: {outputs[1].shape}")
        print(f"  - Text direction: {outputs[2].shape}")
        print(f"  - Rotation angle: {outputs[3].shape}")
