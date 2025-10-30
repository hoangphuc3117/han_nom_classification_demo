"""
Test script for Multi-Task Model
Quick test to verify model architecture and basic functionality
"""

import torch
from multitask_model import (
    MultiTaskResNet50, 
    load_resnet50_weights_to_custom,
    MAIN_CATEGORIES,
    DOC_TYPES,
    TEXT_DIRECTIONS,
    ROTATION_ANGLES,
    IMAGE_SIZE,
    IMAGE_DEPTH
)

def test_model_creation():
    """Test model creation and initialization"""
    print("=" * 60)
    print("🧪 Testing Multi-Task Model Creation")
    print("=" * 60)
    
    try:
        model = MultiTaskResNet50(
            use_cbam=True,
            image_depth=IMAGE_DEPTH,
            num_classes=[len(MAIN_CATEGORIES), len(DOC_TYPES), len(TEXT_DIRECTIONS)],
            num_rotations=len(ROTATION_ANGLES)
        )
        print("✅ Model created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Model Statistics:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return None

def test_pretrained_weights(model):
    """Test loading pretrained ResNet50 weights"""
    print("\n" + "=" * 60)
    print("🧪 Testing Pretrained Weights Loading")
    print("=" * 60)
    
    try:
        load_resnet50_weights_to_custom(model)
        print("✅ Pretrained weights loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading pretrained weights: {str(e)}")
        return False

def test_forward_pass(model):
    """Test forward pass with dummy input"""
    print("\n" + "=" * 60)
    print("🧪 Testing Forward Pass")
    print("=" * 60)
    
    try:
        model.eval()
        
        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, IMAGE_DEPTH, IMAGE_SIZE[0], IMAGE_SIZE[1])
        print(f"📥 Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Check outputs
        expected_shapes = [
            (batch_size, len(MAIN_CATEGORIES)),
            (batch_size, len(DOC_TYPES)),
            (batch_size, len(TEXT_DIRECTIONS)),
            (batch_size, len(ROTATION_ANGLES))
        ]
        
        task_names = ["Main Category", "Document Type", "Text Direction", "Rotation Angle"]
        
        print("📤 Output shapes:")
        all_correct = True
        for i, (output, expected_shape, task_name) in enumerate(zip(outputs, expected_shapes, task_names)):
            is_correct = output.shape == expected_shape
            status = "✅" if is_correct else "❌"
            print(f"   {status} {task_name}: {output.shape} (expected {expected_shape})")
            all_correct = all_correct and is_correct
        
        if all_correct:
            print("\n✅ All output shapes are correct!")
            return True
        else:
            print("\n❌ Some output shapes are incorrect!")
            return False
            
    except Exception as e:
        print(f"❌ Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_logic(model):
    """Test prediction with probabilities"""
    print("\n" + "=" * 60)
    print("🧪 Testing Prediction Logic")
    print("=" * 60)
    
    try:
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, IMAGE_DEPTH, IMAGE_SIZE[0], IMAGE_SIZE[1])
        
        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Convert to probabilities
        import torch.nn.functional as F
        prob_main = F.softmax(outputs[0], dim=1)
        prob_doc = F.softmax(outputs[1], dim=1)
        prob_text = F.softmax(outputs[2], dim=1)
        prob_rotation = F.softmax(outputs[3], dim=1)
        
        print("📊 Predicted Probabilities:")
        print(f"\n   Main Category:")
        for name, prob in zip(MAIN_CATEGORIES.keys(), prob_main[0]):
            print(f"      - {name}: {prob.item():.4f}")
        
        print(f"\n   Document Type:")
        for name, prob in zip(DOC_TYPES.keys(), prob_doc[0]):
            print(f"      - {name}: {prob.item():.4f}")
        
        print(f"\n   Text Direction:")
        for name, prob in zip(TEXT_DIRECTIONS.keys(), prob_text[0]):
            print(f"      - {name}: {prob.item():.4f}")
        
        print(f"\n   Rotation Angle:")
        for angle, prob in zip(ROTATION_ANGLES.keys(), prob_rotation[0]):
            print(f"      - {angle}°: {prob.item():.4f}")
        
        # Get predictions
        pred_main = prob_main.argmax(dim=1).item()
        pred_doc = prob_doc.argmax(dim=1).item()
        pred_text = prob_text.argmax(dim=1).item()
        pred_rotation = prob_rotation.argmax(dim=1).item()
        
        print(f"\n🎯 Final Predictions:")
        print(f"   - Main Category: {list(MAIN_CATEGORIES.keys())[pred_main]} (confidence: {prob_main.max().item():.2%})")
        print(f"   - Document Type: {list(DOC_TYPES.keys())[pred_doc]} (confidence: {prob_doc.max().item():.2%})")
        print(f"   - Text Direction: {list(TEXT_DIRECTIONS.keys())[pred_text]} (confidence: {prob_text.max().item():.2%})")
        print(f"   - Rotation Angle: {list(ROTATION_ANGLES.keys())[pred_rotation]}° (confidence: {prob_rotation.max().item():.2%})")
        
        print("\n✅ Prediction logic test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🏛️ " * 20)
    print("MULTI-TASK MODEL TEST SUITE")
    print("🏛️ " * 20 + "\n")
    
    # Test 1: Model creation
    model = test_model_creation()
    if model is None:
        print("\n❌ Test suite failed at model creation!")
        return
    
    # Test 2: Load pretrained weights
    if not test_pretrained_weights(model):
        print("\n⚠️ Warning: Could not load pretrained weights, but continuing tests...")
    
    # Test 3: Forward pass
    if not test_forward_pass(model):
        print("\n❌ Test suite failed at forward pass!")
        return
    
    # Test 4: Prediction logic
    if not test_prediction_logic(model):
        print("\n❌ Test suite failed at prediction logic!")
        return
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print("\n✅ Multi-Task Model is ready to use!")
    print("\n📝 Next steps:")
    print("   1. Run the Streamlit demo: streamlit run streamlit_multitask_demo.py")
    print("   2. Or use the model in your own code - see README_MULTITASK.md")
    print("\n")

if __name__ == "__main__":
    main()
