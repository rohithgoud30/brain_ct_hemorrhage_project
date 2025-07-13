import streamlit as st
from PIL import Image
import numpy as np
import os

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
        
        # First Conv Block - 16 filters
        layers.Conv2D(16, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block - 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block - 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block - 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Fifth Conv Block - 512 filters
        layers.Conv2D(512, (3, 3), activation='relu', padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
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
    # Define paths for the model files
    keras_model_path = 'models/custom_cnn_best.keras'
    weights_path = 'models/custom_cnn_weights.h5'
    
    # First, try to load the full Keras model
    if os.path.exists(keras_model_path):
        try:
            model = tf.keras.models.load_model(
                keras_model_path,
                compile=False
            )
            st.success("✅ Full model (`.keras`) loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Found `.keras` model but failed to load: {e}")

    # If .keras model fails or doesn't exist, try loading architecture and weights
    if os.path.exists(weights_path):
        try:
            model = create_exact_model_architecture()
            model.load_weights(weights_path)
            st.success("✅ Custom CNN weights (`.h5`) loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load weights: {e}")
            st.warning("⚠️ Using untrained model for demonstration.")
            return create_exact_model_architecture() # Return untrained model
            
    # If neither file is found
    st.error("❌ Model files not found in the 'models/' directory.")
    st.warning("⚠️ Using a fallback, untrained model for demonstration.")
    # Return a simple fallback model if all else fails
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
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add channel dimension: (H, W) -> (H, W, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    
    # Add batch dimension: (H, W, 1) -> (1, H, W, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_hemorrhage(image):
    """Predict if the CT scan shows hemorrhage"""
    try:
        # Get the cached model
        model = get_model()
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0]
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class] * 100)
        
        # Map class to label
        labels = ['Normal', 'Hemorrhage']
        label = labels[predicted_class]
        
        return label, confidence, prediction
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        # Return mock prediction for demo
        return "Unknown", 50.0, np.array([0.5, 0.5])

def main():
    """Main Streamlit app"""
    # Page configuration
    st.set_page_config(
        page_title='Brain Hemorrhage Classification',
        page_icon='🧠',
        layout='centered',
        initial_sidebar_state='collapsed'
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        color: #2E86AB;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .hemorrhage-box {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .model-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.markdown('<div class="main-header">🧠 Brain Hemorrhage Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning CT Scan Classification</div>', unsafe_allow_html=True)
    
    # Model architecture info
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
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a Brain CT Scan Image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Original CT Scan", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Processed Image")
            # Show grayscale version
            gray_image = image.convert('L')
            st.image(gray_image, caption="Grayscale (Model Input)", use_container_width=True)
        
        # Image info
        st.markdown('<div class="model-info">', unsafe_allow_html=True)
        st.write(f"**Image Info:** {image.size[0]}×{image.size[1]} pixels, Mode: {image.mode}")
        st.write(f"**Processed:** Resized to 224×224, Normalized to [0,1]")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🔬 Analyze CT Scan", use_container_width=True):
            with st.spinner("🤖 Analyzing the CT scan..."):
                label, confidence, raw_prediction = predict_hemorrhage(image)
                
                # Display results
                st.markdown("### 📋 Analysis Results")
                
                if label == "Normal":
                    st.markdown(f"""
                    <div class="normal-box">
                        <h2>✅ Normal Brain CT</h2>
                        <p style="font-size: 1.2rem;">No hemorrhage detected</p>
                        <p style="font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif label == "Hemorrhage":
                    st.markdown(f"""
                    <div class="hemorrhage-box">
                        <h2>⚠️ Hemorrhage Detected</h2>
                        <p style="font-size: 1.2rem;">Possible brain hemorrhage found</p>
                        <p style="font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>❓ Analysis Uncertain</h2>
                        <p style="font-size: 1.2rem;">Please consult a medical professional</p>
                        <p style="font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show detailed probabilities
                st.markdown("### 📊 Detailed Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.metric("Normal", f"{raw_prediction[0]*100:.1f}%")
                with prob_col2:
                    st.metric("Hemorrhage", f"{raw_prediction[1]*100:.1f}%")
                
                # Show prediction bar chart
                st.markdown("### 📈 Prediction Visualization")
                chart_data = {
                    'Class': ['Normal', 'Hemorrhage'],
                    'Probability': [raw_prediction[0], raw_prediction[1]]
                }
                st.bar_chart(chart_data, x='Class', y='Probability')
    
    else:
        st.info("👆 Please upload a CT scan image to begin analysis")
        
        # Show some example information
        st.markdown("### 📖 About This Tool")
        st.markdown("""
        This tool uses a trained convolutional neural network to classify brain CT scans. The model attempts 
        to distinguish between normal scans and those that may show hemorrhages based on patterns learned 
        from training data.
        
        **Features:**
        - 🎯 Custom CNN with 5 convolutional blocks
        - 📊 Confidence scores for predictions
        - 🔍 Automatic image preprocessing
        - 💡 Simple interface
        - 📈 Probability visualization
        """)
        
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        1. Upload a brain CT scan image (JPG, PNG, etc.)
        2. Click "Analyze CT Scan" to get predictions
        3. View results with confidence scores
        4. See detailed probability breakdown
        """)

if __name__ == "__main__":
    main()
