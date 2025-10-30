"""
🏛️ Hán Nôm Multi-Task Classification Demo
Streamlit Web Application for Multi-Task Learning

✅ UPDATED: Architecture matches training notebook!

This app demonstrates the HierarchicalResNet50WithRotation model with CBAM for:
- Deep Hierarchical document classification (3 levels with feature concatenation)
- Rotation angle prediction (4 angles: 0°, 90°, 180°, 270°) - Only for SinoNom

Model: HierarchicalResNet50WithRotation
- Backbone: ResNet50 + CBAM attention
- Image size: 128x128
- Training: Conditional model with data augmentation x4
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import io
import kagglehub
from typing import Dict, Tuple, List, Optional, Union
import warnings
import torchvision.models as models
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

# Display mappings for Vietnamese text
DISPLAY_MAIN_CATEGORIES = {
    "SinoNom": "Ảnh Hán Nôm",
    "NonSinoNom": "Ảnh không chứa chữ Hán Nôm"
}

DISPLAY_DOC_TYPES = {
    "Thong_thuong": "Thông thường",
    "Hanh_chinh": "Hành chính", 
    "Ngoai_canh": "Ngoại cảnh"
}

DISPLAY_TEXT_DIRECTIONS = {
    "Doc": "Dọc",
    "Ngang": "Ngang"
}

IMAGE_SIZE = (128, 128)  # ✅ UPDATED: 128x128 for conditional model (matches training)
IMAGE_DEPTH = 3

# ==================== MODEL ARCHITECTURE ====================

class ChannelAttention(nn.Module):
    """Channel Attention in CBAM - UPDATED to match training notebook."""
    
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
    """Merge channels into 2 channels (max and mean)."""
    
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), 
                         torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialAttention(nn.Module):
    """Spatial Attention in CBAM - UPDATED to match training notebook."""
    
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

class CBAM(nn.Module):
    """Convolutional Block Attention Module - UPDATED to match training notebook."""
    
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial
        
        self.channel_attention = ChannelAttention(channel_in, reduction_ratio, pool_types)
        
        if self.spatial:
            self.spatial_attention = SpatialAttention(kernel_size=7)
    
    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)
        return x_out

class BottleneckCBAM(nn.Module):
    """Bottleneck block with CBAM attention - UPDATED to match training notebook."""
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


class HierarchicalResNet50WithRotation(nn.Module):
    """
    ResNet50 with CBAM + Deep Hierarchical Classification + Rotation Detection.
    ✅ UPDATED: Architecture matches training notebook exactly!
    
    Multi-task learning:
    - Shared backbone for feature extraction
    - Hierarchical classification head (3 levels with feature concatenation)
    - Rotation detection head (4 classes: 0°, 90°, 180°, 270°)
    """
    
    def __init__(self, use_cbam=True, image_depth=3, num_classes=[2, 3, 2],
                 enable_rotation=True, num_rotation_classes=4):
        super().__init__()
        self.expansion = 4
        self.use_cbam = use_cbam
        self.enable_rotation = enable_rotation
        self.inplanes = 64
        
        # ==================== SHARED BACKBONE ====================
        self.conv1 = nn.Conv2d(image_depth, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BottleneckCBAM, 64, 3, use_cbam=use_cbam)
        self.layer2 = self._make_layer(BottleneckCBAM, 128, 4, stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(BottleneckCBAM, 256, 6, stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(BottleneckCBAM, 512, 3, stride=2, use_cbam=use_cbam)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ==================== HIERARCHICAL CLASSIFICATION HEAD ====================
        base_feature_dim = 512 * self.expansion  # 2048
        
        # Level 1: Main category (SinoNom/NonSinoNom)
        self.h1_layer = nn.Sequential(
            nn.Linear(base_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Level 2: Document type (depends on h1)
        self.h2_layer = nn.Sequential(
            nn.Linear(base_feature_dim + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Level 3: Text direction (depends on h1 + h2)
        self.h3_layer = nn.Sequential(
            nn.Linear(base_feature_dim + 512 + 256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classification heads
        self.classifier1 = nn.Linear(512, num_classes[0])  # Main category
        self.classifier2 = nn.Linear(256, num_classes[1])  # Document type
        self.classifier3 = nn.Linear(128, num_classes[2])  # Text direction
        
        # ==================== ROTATION DETECTION HEAD ====================
        if self.enable_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(base_feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_rotation_classes)
            )
        
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
        # ==================== SHARED BACKBONE ====================
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        base_features = torch.flatten(x, 1)  # [batch, 2048]
        
        # ==================== HIERARCHICAL CLASSIFICATION ====================
        # Level 1
        h1 = self.h1_layer(base_features)  # [batch, 512]
        out_main = self.classifier1(h1)
        
        # Level 2
        h2_input = torch.cat([base_features, h1], dim=1)
        h2 = self.h2_layer(h2_input)  # [batch, 256]
        out_doc = self.classifier2(h2)
        
        # Level 3
        h3_input = torch.cat([base_features, h1, h2], dim=1)
        h3 = self.h3_layer(h3_input)  # [batch, 128]
        out_text = self.classifier3(h3)
        
        # ==================== ROTATION DETECTION ====================
        if self.enable_rotation:
            out_rotation = self.rotation_head(base_features)
            return [out_main, out_doc, out_text], out_rotation
        else:
            return [out_main, out_doc, out_text]

# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_model():
    """Load the trained Multi-Task model from Kaggle."""
    try:
        # Create model instance - UPDATED to use HierarchicalResNet50WithRotation
        model = HierarchicalResNet50WithRotation(
            use_cbam=True,
            image_depth=IMAGE_DEPTH,
            num_classes=[len(MAIN_CATEGORIES), len(DOC_TYPES), len(TEXT_DIRECTIONS)],
            enable_rotation=True,
            num_rotation_classes=len(ROTATION_ANGLES)
        )

        load_resnet50_weights_to_custom(model)
        
        # Download model from Kaggle
        with st.spinner("Đang tải mô hình Multi-Task từ Kaggle..."):
            try:
                # Download latest version from Kaggle
                path = kagglehub.model_download("phuchoangnguyen/han-nom-classificatiion-multi-task/pyTorch/default")
                
                # Find the model file in the downloaded path
                model_files = [f for f in os.listdir(path) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(path, model_files[0])
                    
                    # Load the model weights with proper error handling
                    try:
                        # Try with weights_only=False directly since we trust Kaggle source
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    except Exception as load_error:
                        st.error(f"❌ Lỗi khi load model: {str(load_error)}")
                        raise load_error
                    
                    # Handle checkpoint format
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Handle DataParallel wrapped models
                    if list(state_dict.keys())[0].startswith('module.'):
                        # Remove 'module.' prefix from keys
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    
                    model.load_state_dict(state_dict)
                    st.success(f"✅ Đã tải model Multi-Task thành công từ Kaggle")
                else:
                    st.warning("⚠️ Không tìm thấy file model trong thư mục tải về. Sử dụng mô hình chưa được huấn luyện.")
                    
            except Exception as e:
                st.error(f"❌ Lỗi khi tải từ Kaggle: {str(e)}")
                st.warning("⚠️ Sử dụng mô hình chưa được huấn luyện (demo architecture)")
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
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

def preprocess_image(image):
    """Preprocess image for model input."""
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
        
        # Define transforms - ✅ UPDATED: Image size is 128x128 for conditional model
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        st.error(f"❌ Lỗi tiền xử lý ảnh: {str(e)}")
        return None

def predict_image(model, image_tensor):
    """Make prediction on preprocessed image with multi-task model."""
    try:
        with torch.no_grad():
            # Model returns tuple: (hierarchical_outputs, rotation_output)
            hierarchical_outputs, rotation_output = model(image_tensor)
            
            # Unpack hierarchical outputs
            out_main, out_doc, out_text = hierarchical_outputs
            
            # Convert to probabilities
            prob_1 = F.softmax(out_main, dim=1)
            prob_2 = F.softmax(out_doc, dim=1)
            prob_3 = F.softmax(out_text, dim=1)
            prob_rotation = F.softmax(rotation_output, dim=1)
            
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
            
            # Convert to display names
            display_main_category = DISPLAY_MAIN_CATEGORIES.get(main_category, main_category)
            display_doc_type = DISPLAY_DOC_TYPES.get(doc_type, doc_type) if doc_type != "N/A" else "N/A"
            display_text_direction = DISPLAY_TEXT_DIRECTIONS.get(text_direction, text_direction) if text_direction != "N/A" else "N/A"
            
            return {
                'main_category': display_main_category,
                'main_category_confidence': conf_1,
                'document_type': display_doc_type,
                'document_type_confidence': conf_2,
                'text_direction': display_text_direction,
                'text_direction_confidence': conf_3,
                'rotation_angle': rotation_angle,
                'rotation_confidence': conf_rotation,
                'raw_probabilities': {
                    'level_1': prob_1.squeeze().tolist(),
                    'level_2': prob_2.squeeze().tolist(),
                    'level_3': prob_3.squeeze().tolist(),
                    'rotation': prob_rotation.squeeze().tolist()
                }
            }
    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {str(e)}")
        return None

def rotate_image(image, angle):
    """Rotate image by specified angle."""
    if angle == 0:
        return image
    elif angle == 90:
        return image.rotate(-90, expand=True)
    elif angle == 180:
        return image.rotate(180, expand=True)
    elif angle == 270:
        return image.rotate(90, expand=True)
    return image

# ==================== STREAMLIT APP ====================

def main():
    # Page config
    st.set_page_config(
        page_title="Hán Nôm Multi-Task Classification Demo",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: black;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        transition: width 0.3s ease;
    }
    .rotation-indicator {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏛️ Hán Nôm Multi-Task Classification Demo</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    ### 🎯 Giới thiệu
    Ứng dụng này sử dụng mô hình **Multi-Task ResNet50 với CBAM** để thực hiện:
    - **Phân loại phân cấp** (Hierarchical Classification): Xác định loại tài liệu Hán Nôm
    - **Dự đoán góc xoay** (Rotation Prediction): Xác định góc xoay của ảnh (0°, 90°, 180°, 270°)
    """)
    
    st.divider()
    
    model = load_model()
    
    if model is None:
        st.error("❌ Không thể tải mô hình. Vui lòng thử lại sau.")
        st.stop()
    
    st.header("📤 Tải ảnh để phân loại")
        
    # File upload
    uploaded_file = st.file_uploader(
        "Chọn hình ảnh tài liệu Hán Nôm:",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Hỗ trợ các định dạng: PNG, JPG, JPEG, TIFF, BMP"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Ảnh gốc")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
            
            # Image info
            st.write(f"**Kích thước:** {image.size}")
            st.write(f"**Định dạng:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        with col2:
            st.subheader("🧠 Kết quả phân loại")
            
            # Automatic prediction when image is uploaded
            with st.spinner("Đang phân tích với Multi-Task Model..."):
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                if image_tensor is not None:
                    # Make prediction
                    prediction = predict_image(model, image_tensor)
                    
                    if prediction:
                        # Rotation Prediction (highlighted section)
                        st.markdown(
                            f'<div class="rotation-indicator">🔄 Góc xoay phát hiện: {prediction["rotation_angle"]}° (Độ tin cậy: {prediction["rotation_confidence"]:.1%})</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Main Category
                        with st.container():
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.metric(
                                    label="📊 Loại chính",
                                    value=prediction['main_category'],
                                    delta=f"Độ tin cậy: {prediction['main_category_confidence']:.1%}"
                                )
                            with col_b:
                                st.progress(prediction['main_category_confidence'])
                        
                        # Document Type  
                        with st.container():
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.metric(
                                    label="📋 Loại tài liệu",
                                    value=prediction['document_type'],
                                    delta=f"Độ tin cậy: {prediction['document_type_confidence']:.1%}" if prediction['document_type'] != "N/A" else "Không áp dụng"
                                )
                            with col_b:
                                if prediction['document_type'] != "N/A":
                                    st.progress(prediction['document_type_confidence'])
                                else:
                                    st.write("—")
                        
                        # Text Direction
                        with st.container():
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.metric(
                                    label="📐 Hướng đọc",
                                    value=prediction['text_direction'],
                                    delta=f"Độ tin cậy: {prediction['text_direction_confidence']:.1%}" if prediction['text_direction'] != "N/A" else "Không áp dụng"
                                )
                            with col_b:
                                if prediction['text_direction'] != "N/A":
                                    st.progress(prediction['text_direction_confidence'])
                                else:
                                    st.write("—")
                        
                        # Show corrected image if rotation detected
                        if prediction['rotation_angle'] != 0:
                            st.divider()
                            st.subheader("✨ Ảnh đã được chỉnh góc")
                            corrected_image = rotate_image(image, prediction['rotation_angle'])
                            st.image(corrected_image, caption=f"Đã xoay {prediction['rotation_angle']}°", use_container_width=True)
                        
                        # Detailed probabilities
                        with st.expander("📈 Chi tiết xác suất"):
                            probs = prediction['raw_probabilities']
                            
                            # Level 1
                            st.write("**Loại chính:**")
                            for i, (name, prob) in enumerate(zip(MAIN_CATEGORIES.keys(), probs['level_1'])):
                                display_name = DISPLAY_MAIN_CATEGORIES.get(name, name)
                                st.write(f"- {display_name}: {prob:.3f}")
                            
                            # Level 2
                            st.write("**Loại tài liệu:**")
                            for i, (name, prob) in enumerate(zip(DOC_TYPES.keys(), probs['level_2'])):
                                display_name = DISPLAY_DOC_TYPES.get(name, name)
                                st.write(f"- {display_name}: {prob:.3f}")
                            
                            # Level 3
                            st.write("**Hướng đọc:**")
                            for i, (name, prob) in enumerate(zip(TEXT_DIRECTIONS.keys(), probs['level_3'])):
                                display_name = DISPLAY_TEXT_DIRECTIONS.get(name, name)
                                st.write(f"- {display_name}: {prob:.3f}")
                            
                            # Rotation
                            st.write("**Góc xoay:**")
                            for i, (angle, prob) in enumerate(zip(ROTATION_ANGLES.keys(), probs['rotation'])):
                                st.write(f"- {angle}°: {prob:.3f}")
                    else:
                        st.error("❌ Không thể thực hiện phân loại")
                else:
                    st.error("❌ Không thể tiền xử lý ảnh")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ Thông tin mô hình")
        st.write("""
        **Kiến trúc:** Multi-Task ResNet50 + CBAM
        
        **Kích thước ảnh:** 224x224
        
        **Tasks:**
        - Hierarchical Classification (3 levels)
        - Rotation Prediction (4 angles)
        
        **Loại chính:**
        - Ảnh Hán Nôm
        - Ảnh không chứa chữ Hán Nôm
        
        **Loại tài liệu:**
        - Thông thường
        - Hành chính
        - Ngoại cảnh
        
        **Hướng đọc:**
        - Dọc
        - Ngang
        
        **Góc xoay:**
        - 0°, 90°, 180°, 270°
        """)
        
        st.divider()
        
        st.write("**Developed by:** Hoang Phuc Nguyen")
        st.write("**Model:** Multi-Task ResNet50 with CBAM")

if __name__ == "__main__":
    main()
