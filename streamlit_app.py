import streamlit as st
from PIL import Image
import numpy as np
import os
from io import BytesIO

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from keras import layers

def create_exact_model_architecture():
    """Create the exact model architecture from your training"""
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2, activation='softmax')
    ], name="custom_cnn")
    
    return model

def load_model():
    """Load the trained model"""
    keras_model_path = 'models/custom_cnn_best.keras'
    weights_path = 'models/custom_cnn_weights.h5'
    
    if os.path.exists(keras_model_path):
        try:
            model = tf.keras.models.load_model(keras_model_path, compile=False)
            st.success("✅ Full model (`.keras`) loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Found `.keras` model but failed to load: {e}")

    if os.path.exists(weights_path):
        try:
            model = create_exact_model_architecture()
            model.load_weights(weights_path)
            st.success("✅ Custom CNN weights (`.h5`) loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load weights: {e}")
            st.warning("⚠️ Using untrained model for demonstration.")
            return create_exact_model_architecture()
            
    st.error("❌ Model files not found in the 'models/' directory.")
    st.warning("⚠️ Using a fallback, untrained model for demonstration.")
    return keras.Sequential([
        layers.Input(shape=(224, 224, 1)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax', name="fallback_output")
    ])

@st.cache_resource
def get_model():
    """Cache the model to avoid reloading"""
    return load_model()

def preprocess_image(image):
    """Preprocess the image for prediction"""
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_hemorrhage(image):
    """Predict if the CT scan shows hemorrhage"""
    try:
        model = get_model()
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class] * 100)
        labels = ['Normal', 'Hemorrhage']
        label = labels[predicted_class]
        
        return label, confidence, prediction
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return "Unknown", 50.0, np.array([0.5, 0.5])

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title='Brain Hemorrhage Classification',
        page_icon='🧠',
        layout='centered',
        initial_sidebar_state='collapsed'
    )
    
    st.markdown("""
    <style>
    .main-header { text-align: center; font-size: 3rem; color: #2E86AB; font-weight: bold; margin-bottom: 0.5rem; }
    .sub-header { text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .prediction-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0; }
    .normal-box { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0; }
    .hemorrhage-box { background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0; }
    .model-info { background: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #2E86AB; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🧠 Brain Hemorrhage Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning CT Scan Classification</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Model Architecture", expanded=False):
        st.markdown("""
        **Custom CNN Architecture:**
        - Input: 224×224 grayscale images
        - 5 Convolutional blocks with BatchNormalization
        - Filters: 16 → 32 → 64 → 128 → 512
        - 3 Dense layers: 1024 → 256 → 128 → 2 (output)
        - Binary classification: Normal vs Hemorrhage
        """)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload a Brain CT Scan Image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            st.error(f"❌ Error opening image file: {e}")
            return

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Original CT Scan", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Processed Image")
            processed_display = preprocess_image(image.copy())
            processed_display = np.squeeze(processed_display)
            st.image(processed_display, caption="Grayscale 224x224", use_container_width=True)

        st.markdown("---")

        with st.spinner("🧠 Analyzing image..."):
            label, confidence, prediction = predict_hemorrhage(image.copy())
        
        if label == "Hemorrhage":
            st.markdown(f'<div class="hemorrhage-box"><h3>Prediction: Hemorrhage</h3><p>Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
        elif label == "Normal":
            st.markdown(f'<div class="normal-box"><h3>Prediction: Normal</h3><p>Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-box"><h3>Prediction: {label}</h3><p>Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)

        with st.expander("🔬 View Prediction Probabilities"):
            st.write({
                "Normal": f"{prediction[0]*100:.2f}%",
                "Hemorrhage": f"{prediction[1]*100:.2f}%"
            })

if __name__ == '__main__':
    main()