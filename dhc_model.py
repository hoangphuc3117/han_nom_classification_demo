"""
DHC Model Architecture without Streamlit dependencies
Pure PyTorch implementation for testing and inference
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

# Reverse mappings for inference
INV_MAIN_CATEGORIES = {v: k for k, v in MAIN_CATEGORIES.items()}
INV_DOC_TYPES = {v: k for k, v in DOC_TYPES.items()}
INV_TEXT_DIRECTIONS = {v: k for k, v in TEXT_DIRECTIONS.items()}

IMAGE_SIZE = (128, 128)
IMAGE_DEPTH = 3

# ==================== MODEL ARCHITECTURE ====================

class ChannelAttention(nn.Module):
    """Channel Attention trong CBAM."""
    
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.pool_types = pool_types
        
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_in, channel_in // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in // reduction_ratio, channel_in)
        )
    
    def forward(self, x):
        channel_attentions = []
        
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), 
                                       stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_type == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), 
                                       stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))
        
        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = torch.sigmoid(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scaled

class ChannelPool(nn.Module):
    """Merge channels thành 2 channels (max và mean)."""
    
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), 
                         torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialAttention(nn.Module):
    """Spatial Attention trong CBAM."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, 
                     padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        )
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = torch.sigmoid(x_output)
        return x * scaled

# ==================== BOTTLENECK BLOCK VÀ RESNET50 ====================

class BottleneckCBAM(nn.Module):
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

class HierarchicalResNet50(nn.Module):
    def __init__(self, use_cbam=True, image_depth=3, num_classes=[2,3,2]):
        super().__init__()
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
        self.fc_main = nn.Linear(2048, num_classes[0])
        self.fc_doc = nn.Linear(2048, num_classes[1])
        self.fc_text = nn.Linear(2048, num_classes[2])
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
        out_main = self.fc_main(x)
        out_doc = self.fc_doc(x)
        out_text = self.fc_text(x)
        return [out_main, out_doc, out_text]
# ==================== UTILITY FUNCTIONS ====================

def load_model_standalone(model_path=None):
    """Load the trained DHC model without Streamlit dependencies."""
    try:
        # Create model instance
        model = HierarchicalResNet50(
            use_cbam=True,
            image_depth=IMAGE_DEPTH,
            num_classes=[len(MAIN_CATEGORIES), len(DOC_TYPES), len(TEXT_DIRECTIONS)]
        )
        
        load_resnet50_weights_to_custom(model)

        # Try to load pre-trained weights if path provided
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Đã tải mô hình từ: {model_path}")
        else:
            print("⚠️ Sử dụng mô hình chưa được huấn luyện (demo architecture)")
        
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return None
    
def load_resnet50_weights_to_custom(model, imagenet_weights=None):
    """
    Load pretrained weights từ torchvision ResNet50 vào custom ResNet50 + CBAM.
    """
    if imagenet_weights is None:
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet50 = models.resnet50()
        resnet50.load_state_dict(torch.load(imagenet_weights))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in resnet50.state_dict().items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded pretrained weights cho backbone custom ResNet50 + CBAM!")

def preprocess_image_standalone(image):
    """Preprocess image for model input without Streamlit dependencies."""
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
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"❌ Lỗi tiền xử lý ảnh: {str(e)}")
        return None

def predict_image_standalone(model, image_tensor):
    """Make prediction on preprocessed image without Streamlit dependencies."""
    try:
        with torch.no_grad():
            predictions = model(image_tensor)
            
            # Convert to probabilities
            prob_1 = F.softmax(predictions[0], dim=1)
            prob_2 = F.softmax(predictions[1], dim=1)
            prob_3 = F.softmax(predictions[2], dim=1)
            
            # Get predicted classes
            pred_1 = prob_1.argmax(dim=1).item()
            pred_2 = prob_2.argmax(dim=1).item()
            pred_3 = prob_3.argmax(dim=1).item()
            
            # Get confidence scores
            conf_1 = prob_1.max().item()
            conf_2 = prob_2.max().item()
            conf_3 = prob_3.max().item()
            
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
            
            return {
                'main_category': main_category,
                'main_category_confidence': conf_1,
                'document_type': doc_type,
                'document_type_confidence': conf_2,
                'text_direction': text_direction,
                'text_direction_confidence': conf_3,
                'raw_probabilities': {
                    'level_1': prob_1.squeeze().tolist(),
                    'level_2': prob_2.squeeze().tolist(),
                    'level_3': prob_3.squeeze().tolist()
                }
            }
    except Exception as e:
        print(f"❌ Lỗi dự đoán: {str(e)}")
        return None
