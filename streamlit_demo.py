"""
🏛️ Hán Nôm Classification Demo with DHC Model
Streamlit Web Application for Hierarchical Document Classification

This app demonstrates the Deep Hierarchical Classification (DHC) model
for classifying Han-Nom documents.
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

# WebRTC for real-time camera
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("⚠️ streamlit-webrtc chưa được cài đặt. Để sử dụng camera real-time, chạy: pip install streamlit-webrtc")

# ==================== CONSTANTS ====================
MAIN_CATEGORIES = {"SinoNom": 0, "NonSinoNom": 1}
DOC_TYPES = {"Thong_thuong": 0, "Hanh_chinh": 1, "Ngoai_canh": 2}
TEXT_DIRECTIONS = {"Doc": 0, "Ngang": 1}

# Reverse mappings for inference
INV_MAIN_CATEGORIES = {v: k for k, v in MAIN_CATEGORIES.items()}
INV_DOC_TYPES = {v: k for k, v in DOC_TYPES.items()}
INV_TEXT_DIRECTIONS = {v: k for k, v in TEXT_DIRECTIONS.items()}

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

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
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

@st.cache_resource
def load_model():
    """Load the trained DHC model from Kaggle."""
    try:
        # Create model instance
        model = HierarchicalResNet50(
            use_cbam=True,
            image_depth=IMAGE_DEPTH,
            num_classes=[len(MAIN_CATEGORIES), len(DOC_TYPES), len(TEXT_DIRECTIONS)]
        )

        load_resnet50_weights_to_custom(model)
        
        # Download model from Kaggle
        with st.spinner("Đang tải mô hình từ Kaggle..."):
            try:
                # Download latest version from Kaggle
                path = kagglehub.model_download("phuchoangnguyen/han-nom-classification/pyTorch/default")
                
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
                    
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    st.success(f"✅ Đã tải model thành công")
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
        # Only use st.error if in Streamlit context
        try:
            st.error(f"❌ Lỗi tiền xử lý ảnh: {str(e)}")
        except:
            print(f"❌ Lỗi tiền xử lý ảnh: {str(e)}")
        return None

def predict_image(model, image_tensor):
    """Make prediction on preprocessed image."""
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
                'raw_probabilities': {
                    'level_1': prob_1.squeeze().tolist(),
                    'level_2': prob_2.squeeze().tolist(),
                    'level_3': prob_3.squeeze().tolist()
                }
            }
    except Exception as e:
        # Only use st.error if in Streamlit context
        try:
            st.error(f"❌ Lỗi dự đoán: {str(e)}")
        except:
            print(f"❌ Lỗi dự đoán: {str(e)}")
        return None

# ==================== VIDEO PROCESSOR FOR WEBRTC ====================

if WEBRTC_AVAILABLE:
    class VideoProcessor(VideoProcessorBase):
        """Video processor for real-time classification"""
        
        def __init__(self):
            self.model = None
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.prediction_text = "Đang chờ..."
            self.frame_count = 0
            self.process_every_n_frames = 10  # Chỉ xử lý mỗi 10 frames để tăng tốc độ
        
        def set_model(self, model):
            """Set the model for inference"""
            self.model = model
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """Process each video frame"""
            try:
                # Convert frame to numpy array
                img = frame.to_ndarray(format="bgr24")
                
                # Process every N frames
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames == 0 and self.model is not None:
                    try:
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Preprocess
                        img_tensor = self.transform(img_rgb).unsqueeze(0)
                        
                        # Predict
                        with torch.no_grad():
                            predictions = self.model(img_tensor)
                            
                            # Get probabilities
                            prob_1 = F.softmax(predictions[0], dim=1)
                            prob_2 = F.softmax(predictions[1], dim=1)
                            prob_3 = F.softmax(predictions[2], dim=1)
                            
                            # Get predictions
                            pred_1 = prob_1.argmax(dim=1).item()
                            pred_2 = prob_2.argmax(dim=1).item()
                            pred_3 = prob_3.argmax(dim=1).item()
                            
                            # Get confidence
                            conf_1 = prob_1.max().item()
                            
                            # Apply hierarchical logic
                            main_category = INV_MAIN_CATEGORIES[pred_1]
                            display_main = DISPLAY_MAIN_CATEGORIES.get(main_category, main_category)
                            
                            if main_category == "SinoNom":
                                doc_type = INV_DOC_TYPES[pred_2]
                                display_doc = DISPLAY_DOC_TYPES.get(doc_type, doc_type)
                                
                                if doc_type == "Thong_thuong":
                                    text_direction = INV_TEXT_DIRECTIONS[pred_3]
                                    display_text = DISPLAY_TEXT_DIRECTIONS.get(text_direction, text_direction)
                                    self.prediction_text = f"{display_main} | {display_doc} | {display_text} ({conf_1:.1%})"
                                else:
                                    self.prediction_text = f"{display_main} | {display_doc} ({conf_1:.1%})"
                            else:
                                self.prediction_text = f"{display_main} ({conf_1:.1%})"
                    
                    except Exception as e:
                        self.prediction_text = f"Lỗi: {str(e)[:50]}"
                
                # Draw prediction text on frame (with background for better visibility)
                text = self.prediction_text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Draw black background for text
                cv2.rectangle(img, (5, 5), (text_size[0] + 15, 40), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(
                    img, 
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Draw frame count
                cv2.putText(
                    img,
                    f"Frame: {self.frame_count}",
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                
                # Return the processed frame
                return av.VideoFrame.from_ndarray(img, format="bgr24")
                
            except Exception as e:
                # If anything fails, just return the original frame
                print(f"Error in recv: {e}")
                return frame

# ==================== STREAMLIT APP ====================

def main():
    # Page config
    st.set_page_config(
        page_title="Hán Nôm Classification Demo",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏛️ Hán Nôm Classification Demo</h1>', unsafe_allow_html=True)
    
    model = load_model()
    
    if model is None:
        st.error("❌ Không thể tải mô hình. Vui lòng thử lại sau.")
        st.stop()
    
    # Create tabs for different input methods
    if WEBRTC_AVAILABLE:
        tab1, tab2 = st.tabs(["📤 Upload Ảnh", "📹 Camera Real-time"])
    else:
        tab1 = st.container()
        tab2 = None
    
    # Tab 1: File Upload
    with tab1 if WEBRTC_AVAILABLE else tab1:
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
                st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=False)
                
                # Image info
                st.write(f"**Kích thước:** {image.size}")
                st.write(f"**Định dạng:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
            
            with col2:
                st.subheader("🧠 Kết quả phân loại")
                
                # Automatic prediction when image is uploaded
                with st.spinner("Đang phân tích..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    if image_tensor is not None:
                        # Make prediction
                        prediction = predict_image(model, image_tensor)
                        
                        if prediction:    
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
                        else:
                            st.error("❌ Không thể thực hiện phân loại")
                    else:
                        st.error("❌ Không thể tiền xử lý ảnh")
    
    # Tab 2: Camera Real-time
    if WEBRTC_AVAILABLE and tab2 is not None:
        with tab2:
            st.header("📹 Camera Real-time Classification")
            st.write("""
            ### Hướng dẫn sử dụng:
            1. Nhấn nút **START** để bật camera
            2. Hướng camera về tài liệu Hán Nôm
            3. Kết quả phân loại sẽ hiển thị trực tiếp trên video
            4. Nhấn **STOP** để dừng camera
            
            **Lưu ý:** 
            - Kết quả được cập nhật mỗi 10 frames để đảm bảo hiệu suất
            - Độ chính xác tốt nhất khi ảnh rõ nét và đủ ánh sáng
            """)
            
            # Info boxes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("💡 **Tip:** Giữ camera ổn định")
            with col2:
                st.info("🔆 **Ánh sáng:** Đảm bảo đủ sáng")
            with col3:
                st.info("📏 **Khoảng cách:** Không quá xa/gần")
            
            st.divider()
            
            # Factory function to create video processor
            def video_processor_factory():
                processor = VideoProcessor()
                processor.set_model(model)
                return processor
            
            # WebRTC configuration with better STUN/TURN servers
            RTC_CONFIGURATION = {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            }
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="han-nom-classifier",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=video_processor_factory,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                    },
                    "audio": False
                },
                async_processing=True,
                rtc_configuration=RTC_CONFIGURATION,
            )
            
            # Display connection status
            st.divider()
            
            # Debug information
            with st.expander("🔍 Thông tin kết nối (Debug)", expanded=False):
                st.write(f"**Playing:** {webrtc_ctx.state.playing}")
                st.write(f"**Signalling State:** {webrtc_ctx.state.signalling}")
                if webrtc_ctx.video_processor:
                    st.write("✅ Video processor đã được tạo")
                else:
                    st.write("❌ Video processor chưa được tạo")
            
            # Display status
            if webrtc_ctx.state.playing:
                st.success("✅ Camera đang hoạt động - Đang phân loại real-time...")
                
                # Display current prediction from the active processor
                if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'prediction_text'):
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 10px;
                        color: white;
                        text-align: center;
                        font-size: 1.2rem;
                        font-weight: bold;
                        margin: 20px 0;
                    ">
                        🎯 Kết quả hiện tại: {webrtc_ctx.video_processor.prediction_text}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Nhấn START để bắt đầu camera")
            
            # Tips
            with st.expander("📖 Các mẹo để có kết quả tốt nhất"):
                st.write("""
                - **Giữ camera ổn định**: Tránh rung lắc để model dễ nhận diện
                - **Ánh sáng tốt**: Đảm bảo tài liệu được chiếu sáng đều
                - **Khoảng cách phù hợp**: Tài liệu nên chiếm khoảng 70-80% khung hình
                - **Góc nhìn thẳng**: Tránh chụp nghiêng quá nhiều
                - **Chất lượng ảnh**: Camera có độ phân giải tốt sẽ cho kết quả chính xác hơn
                """)

if __name__ == "__main__":
    main()

