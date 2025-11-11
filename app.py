# Force NumPy 1.x compatibility
import os
os.environ["NPY_NO_DEPRECATED_API"] = "NPY_1_7_API_VERSION"

import streamlit as st
import numpy as np

# Check NumPy version compatibility
try:
    # Force numpy to use legacy mode if needed
    if hasattr(np, '_NoValue'):
        np._NoValue = np._globals._NoValue if hasattr(np, '_globals') else None
except:
    pass

import matplotlib.pyplot as plt
from PIL import Image

# Try to import EasyOCR with fallback to Tesseract
try:
    import easyocr
    OCR_ENGINE = "easyocr"
except ImportError as e:
    st.warning(f"EasyOCR not available ({e}). Using Tesseract fallback.")
    try:
        import pytesseract
        OCR_ENGINE = "tesseract" 
    except ImportError:
        st.error("No OCR engine available!")
        st.stop()

from ultralytics import YOLO
import time

# Import GTSRB classes for sign classification
try:
    from gtsrb_classes import GTSRB_CLASSES, get_sign_meaning, get_sign_category
except ImportError:
    # Fallback GTSRB classes if file not found
    GTSRB_CLASSES = [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
        'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
        'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
        'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry',
        'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road',
        'Slippery road', 'Road narrows right', 'Road work', 'Traffic signals', 'Pedestrians',
        'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
        'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
        'End of no passing', 'End no passing veh > 3.5 tons'
    ]
    
    def get_sign_meaning(class_id):
        if 0 <= class_id < len(GTSRB_CLASSES):
            return GTSRB_CLASSES[class_id]
        return f"Unknown class {class_id}"
    
    def get_sign_category(class_id):
        if 0 <= class_id <= 8:
            return "🚫 Speed Limits"
        elif class_id == 14:
            return "🛑 Stop Signs"
        elif class_id == 13:
            return "⚠️ Yield Signs"
        elif 9 <= class_id <= 17:
            return "🚫 Prohibition Signs"
        elif 18 <= class_id <= 31:
            return "⚠️ Warning Signs"
        elif 32 <= class_id <= 42:
            return "➡️ Mandatory Signs"
        else:
            return "🚦 Traffic Signs"

# Page config
st.set_page_config(
    page_title="🚗 Street Sign OCR - AI for Autonomous Driving",
    page_icon="🚦",
    layout="wide"
)

# IMPROVED CSS with better visibility and contrast
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.detection-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #dee2e6;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.sign-classification {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border: 2px solid #2196F3;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    color: #0d47a1;
}

.sign-classification h4 {
    margin: 0;
    color: #1976D2 !important;
    font-weight: bold;
}

.sign-classification p {
    margin: 0.25rem 0 0 0;
    color: #424242 !important;
    font-size: 0.9em;
}

.ocr-result {
    background: linear-gradient(135deg, #f1f8e9 0%, #dcedc1 100%);
    border: 2px solid #4caf50;
    border-radius: 5px;
    padding: 0.5rem;
    margin: 0.25rem 0;
    color: #1b5e20;
}

.ocr-result strong {
    color: #2e7d32 !important;
}

.debug-info {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border: 1px solid #ff9800;
    border-radius: 6px;
    padding: 0.5rem;
    margin: 0.25rem 0;
    font-family: monospace;
    color: #e65100;
}

.info-box {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    border-left: 4px solid #4caf50;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #1b5e20;
}

.warning-box {
    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    border-left: 4px solid #ff9800;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #e65100;
}

.error-box {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 4px solid #f44336;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #c62828;
}

/* Better text visibility */
.stMarkdown p, .stMarkdown div {
    color: #212529 !important;
}

/* Improve sidebar styling */
.css-1d391kg {
    background-color: #f8f9fa;
}

/* Better button styling */
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Improve upload area */
.css-1kyxreq {
    border: 2px dashed #667eea;
    border-radius: 10px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load YOLO and OCR models with enhanced debugging"""
    with st.spinner("🤖 Loading AI models... This may take 1-2 minutes on first run."):
        try:
            # Try to load your custom trained YOLO model first
            detector = None
            model_type = "generic"
            
            try:
                # Check if best.pt exists and get file info
                if os.path.exists('best.pt'):
                    file_size = os.path.getsize('best.pt') / (1024*1024)  # MB
                    st.info(f"📁 Found best.pt ({file_size:.1f} MB)")
                    
                    detector = YOLO('best.pt')  # Your trained GTSRB model
                    
                    # Verify the model has the expected number of classes
                    model_classes = detector.model.names if hasattr(detector.model, 'names') else {}
                    num_classes = len(model_classes)
                    
                    if num_classes == 43:
                        st.success(f"✅ **PERFECT!** Custom GTSRB model loaded with {num_classes} classes!")
                        model_type = "custom"
                    elif num_classes == 1:
                        st.error(f"❌ **TRAINING ISSUE:** Model has only 1 class instead of 43!")
                        st.error("This is the single-class bug. Use the corrected training notebook.")
                        model_type = "single_class"
                    else:
                        st.warning(f"⚠️ Unexpected: Model has {num_classes} classes (expected 43)")
                        model_type = "custom"
                    
                    # Show first few class names for verification
                    if model_classes:
                        sample_classes = list(model_classes.values())[:5]
                        st.info(f"📋 Sample classes: {', '.join(sample_classes)}")
                else:
                    st.warning("📁 No best.pt found - will use generic YOLO")
                    
            except Exception as e:
                st.error(f"❌ Failed to load best.pt: {str(e)}")
                detector = None
            
            # Fallback to generic YOLO if custom model failed
            if detector is None:
                detector = YOLO('yolov8n.pt')
                st.warning("⚠️ Using generic YOLO - Upload your trained 'best.pt' for sign classification")
                model_type = "generic"
            
            # Load OCR model based on available engine
            ocr_reader = None
            if OCR_ENGINE == "easyocr":
                try:
                    ocr_reader = easyocr.Reader(['en', 'de'], gpu=False)
                    st.success("✅ EasyOCR model loaded!")
                except Exception as e:
                    st.error(f"EasyOCR loading failed: {e}")
                    
            elif OCR_ENGINE == "tesseract":
                ocr_reader = None  # Tesseract doesn't need pre-loading
                st.success("✅ Tesseract OCR ready!")
            else:
                st.error("No OCR engine available!")
                return None, None, None
            
            return detector, ocr_reader, model_type
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.error("Please try refreshing the page or contact support.")
            return None, None, None

def preprocess_for_ocr(image):
    """Enhanced preprocessing for better OCR (without cv2)"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure proper format
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        # Simple resize for small images
        height, width = image.shape[:2] if len(image.shape) > 2 else image.shape
        if height < 100 or width < 100:
            scale = max(150/height, 150/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            # Use PIL for resizing instead of cv2
            if len(image.shape) == 3:
                img_pil = Image.fromarray(image)
            else:
                img_pil = Image.fromarray(image, mode='L')
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image = np.array(img_resized)
        
        return image
        
    except Exception:
        return image

def process_uploaded_image(uploaded_file, detector, ocr_reader, confidence_threshold, model_type):
    """Process uploaded image through AI pipeline with enhanced debugging"""
    results = {'detections': [], 'processing_time': 0, 'model_type': model_type}
    
    try:
        start_time = time.time()
        
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        st.info(f"🖼️ Processing image: {image.size[0]}x{image.size[1]} pixels")
        
        # YOLO detection with multiple confidence levels for debugging
        confidence_levels = [confidence_threshold, 0.1, 0.05, 0.01]
        all_detections = []
        
        for conf_level in confidence_levels:
            try:
                detection_results = detector(image_np, conf=conf_level)
                
                # Process each detection
                for result in detection_results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        st.success(f"🎯 Found {len(boxes)} detection(s) at confidence {conf_level}")
                        
                        for box in boxes:
                            # Extract box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            # Skip if we already processed this detection at higher confidence
                            if confidence >= confidence_threshold:
                                # Get class ID and sign classification
                                class_id = int(box.cls[0]) if box.cls is not None else -1
                                
                                # Determine sign meaning based on model type
                                if model_type == "custom" or model_type == "single_class":
                                    # Use GTSRB classification for trained model
                                    sign_meaning = get_sign_meaning(class_id)
                                    sign_category = get_sign_category(class_id)
                                else:
                                    # Generic YOLO classes
                                    generic_classes = {0: 'person', 1: 'bicycle', 2: 'car', 9: 'traffic light', 11: 'stop sign'}
                                    sign_meaning = generic_classes.get(class_id, f"Object (class {class_id})")
                                    sign_category = "🔍 Generic Detection"
                                
                                # Extract ROI for OCR
                                roi = image_np[y1:y2, x1:x2]
                                
                                # Perform OCR based on available engine
                                ocr_results = []
                                if roi.size > 0:
                                    try:
                                        if OCR_ENGINE == "easyocr" and ocr_reader:
                                            # Preprocess ROI
                                            processed_roi = preprocess_for_ocr(roi)
                                            
                                            # Run OCR on both original and processed ROI
                                            raw_ocr = ocr_reader.readtext(roi, paragraph=False)
                                            processed_ocr = ocr_reader.readtext(processed_roi, paragraph=False)
                                            
                                            # Combine and filter results
                                            all_ocr = raw_ocr + processed_ocr
                                            seen_texts = set()
                                            
                                            for (bbox, text, conf) in all_ocr:
                                                clean_text = text.strip()
                                                if conf > 0.3 and clean_text and clean_text not in seen_texts:
                                                    ocr_results.append({
                                                        'text': clean_text,
                                                        'confidence': float(conf)
                                                    })
                                                    seen_texts.add(clean_text)
                                        
                                        elif OCR_ENGINE == "tesseract":
                                            # Use Tesseract OCR
                                            roi_pil = Image.fromarray(roi)
                                            
                                            # OCR with confidence
                                            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                                            data = pytesseract.image_to_data(roi_pil, config=config, output_type=pytesseract.Output.DICT)
                                            
                                            seen_texts = set()
                                            for i, text in enumerate(data['text']):
                                                if text.strip() and int(data['conf'][i]) > 30:
                                                    clean_text = text.strip()
                                                    if clean_text not in seen_texts:
                                                        ocr_results.append({
                                                            'text': clean_text,
                                                            'confidence': int(data['conf'][i]) / 100.0
                                                        })
                                                        seen_texts.add(clean_text)
                                        
                                        # Sort by confidence
                                        ocr_results.sort(key=lambda x: x['confidence'], reverse=True)
                                        
                                    except Exception as ocr_error:
                                        st.warning(f"OCR processing issue: {str(ocr_error)}")
                                
                                results['detections'].append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(confidence),
                                    'class_id': class_id,
                                    'sign_meaning': sign_meaning,
                                    'sign_category': sign_category,
                                    'ocr_results': ocr_results
                                })
                        break  # Found detections, stop trying lower confidence levels
                    else:
                        st.warning(f"⚠️ No detections found at confidence {conf_level}")
                        
            except Exception as e:
                st.error(f"Detection error at confidence {conf_level}: {str(e)}")
                continue
        
        results['processing_time'] = time.time() - start_time
        
        if not results['detections']:
            st.error("❌ No traffic signs detected at any confidence level")
            st.info("💡 Try: Lower confidence threshold, better lighting, or clearer image")
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    
    return results

def create_visualization(image, results):
    """Create detection visualization with matplotlib"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Display image
    ax.imshow(image)
    
    # Color scheme for bounding boxes
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#44FFFF']
    
    # Draw each detection
    for i, detection in enumerate(results['detections']):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        sign_meaning = detection.get('sign_meaning', 'Unknown')
        sign_category = detection.get('sign_category', '🔍 Detection')
        color = colors[i % len(colors)]
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color=color, linewidth=3)
        ax.add_patch(rect)
        
        # Prepare enhanced text annotation
        ocr_texts = [r['text'] for r in detection['ocr_results'][:2]]
        
        # Create comprehensive annotation
        if results.get('model_type') in ['custom', 'single_class']:
            annotation = f"{sign_category}\n{sign_meaning}\nConf: {confidence:.2f}"
        else:
            annotation = f"{sign_meaning}\nConf: {confidence:.2f}"
            
        if ocr_texts:
            annotation += f"\nText: {', '.join(ocr_texts)}"
        
        # Add text with background
        ax.text(x1, y1-15, annotation, 
                bbox=dict(facecolor='white', alpha=0.95, edgecolor=color, pad=5),
                fontsize=9, fontweight='bold', verticalalignment='top')
    
    # Enhanced title
    if results.get('model_type') == 'custom':
        model_status = "🎯 GTSRB Trained Model (43 Classes)"
    elif results.get('model_type') == 'single_class':
        model_status = "⚠️ GTSRB Model (1 Class - Needs Retraining)"
    else:
        model_status = "⚠️ Generic YOLO"
        
    ax.set_title(f"{model_status}: {len(results['detections'])} Signs Detected", 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚗🚦 Street Sign OCR for Autonomous Driving</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🤖 AI-Powered Traffic Sign Detection & Recognition</h3>
    <p><strong>Upload any street image</strong> and watch our AI detect traffic signs and read their text!</p>
    <p>🔬 <em>Built with YOLOv8 + EasyOCR • Perfect for Autonomous Driving Research</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    detector, ocr_reader, model_type = load_models()
    
    if detector is None:
        st.markdown("""
        <div class="error-box">
        <h4>❌ Failed to load AI models</h4>
        <p>Please try refreshing the page. If the issue persists, the models may be downloading.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ AI Settings")
    confidence_threshold = st.sidebar.slider(
        "🎯 Detection Confidence", 
        min_value=0.01, max_value=1.0, value=0.15, step=0.01,
        help="Lower values detect more objects but may include false positives"
    )
    
    # Enhanced debugging toggle
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=True, help="Show detailed detection information")
    
    st.sidebar.markdown("## 📊 System Info")
    if model_type == "custom":
        st.sidebar.markdown("""
        <div class="info-box">
        <strong>🤖 AI Models:</strong><br>
        • YOLOv8: 43-class GTSRB model ✅<br>
        • EasyOCR: Text recognition (EN/DE)<br><br>
        <strong>⚡ Performance:</strong><br>
        • Multi-class detection<br>
        • Real-time processing<br>
        • Cloud-optimized
        </div>
        """, unsafe_allow_html=True)
    elif model_type == "single_class":
        st.sidebar.markdown("""
        <div class="warning-box">
        <strong>⚠️ Single-Class Model:</strong><br>
        • YOLOv8: Only 1 class (needs retraining)<br>
        • EasyOCR: Text recognition (EN/DE)<br><br>
        <strong>🔧 Fix Needed:</strong><br>
        • Use corrected training notebook<br>
        • Retrain for 43 classes
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="warning-box">
        <strong>⚠️ Generic Model:</strong><br>
        • YOLOv8: Generic object detection<br>
        • EasyOCR: Text recognition (EN/DE)<br><br>
        <strong>📁 Upload best.pt:</strong><br>
        • For traffic sign classification<br>
        • 43 GTSRB classes
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("## 🎯 How It Works")
    st.sidebar.info("""
    1. **Upload** your street image
    2. **YOLO** finds traffic signs
    3. **OCR** reads sign text
    4. **View** results with confidence scores
    """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose a street image containing traffic signs...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            # Display image info
            st.markdown(f"""
            <div class="debug-info">
            📐 Image size: {image.size[0]} × {image.size[1]} pixels<br>
            📁 File size: {len(uploaded_file.getvalue()) / 1024:.1f} KB<br>
            🎨 Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🤖 AI Analysis")
        
        if uploaded_file:
            with st.spinner("🔄 AI is analyzing your image... Please wait."):
                results = process_uploaded_image(uploaded_file, detector, ocr_reader, confidence_threshold, model_type)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-container">
                <h3>🎯 {len(results['detections'])}</h3>
                <p>Signs Detected</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                text_count = sum(1 for d in results['detections'] if d['ocr_results'])
                st.markdown(f"""
                <div class="metric-container">
                <h3>📝 {text_count}</h3>
                <p>With Text</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-container">
                <h3>⚡ {results['processing_time']:.1f}s</h3>
                <p>Process Time</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <h4>👆 Upload an image above to see AI analysis results!</h4>
            <p>Try images with clear traffic signs for best results.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Results section
    if uploaded_file and results['detections']:
        st.markdown("---")
        st.markdown("### 🎯 Detection Results")
        
        # Show model status with enhanced styling
        if results.get('model_type') == "custom":
            st.markdown("""
            <div class="info-box">
            <h4>✅ Using Trained GTSRB Model</h4>
            <p>Will classify specific traffic signs (Stop, Speed Limit, etc.)</p>
            </div>
            """, unsafe_allow_html=True)
        elif results.get('model_type') == "single_class":
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Single-Class Model Detected</h4>
            <p>This model only predicts one class. Use the corrected training notebook to fix this.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Using Generic YOLO Model</h4>
            <p>Shows bounding boxes only. Upload your trained model for sign classification!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show visualization
        fig = create_visualization(np.array(image), results)
        st.pyplot(fig, use_container_width=True)
        
        # Detailed results
        st.markdown("### 📋 Detailed Analysis")
        
        for i, detection in enumerate(results['detections']):
            with st.expander(f"🚦 Sign {i+1} - Confidence: {detection['confidence']:.3f}", expanded=True):
                
                detail_col1, detail_col2 = st.columns([1, 2])
                
                with detail_col1:
                    st.markdown("**📍 Detection Info:**")
                    st.markdown(f"""
                    <div class="detection-card">
                    • <strong>Confidence:</strong> {detection['confidence']:.3f}<br>
                    • <strong>Location:</strong> {detection['bbox']}<br>
                    • <strong>Class ID:</strong> {detection.get('class_id', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show traffic sign classification if available
                    if 'class_id' in detection and detection['class_id'] is not None:
                        sign_meaning = detection['sign_meaning']
                        sign_category = detection['sign_category']
                        
                        st.markdown(f"""
                        <div class="sign-classification">
                        <h4>🚦 {sign_meaning}</h4>
                        <p>Category: {sign_category}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional debug for GTSRB classes
                        if debug_mode:
                            if model_type == "custom":
                                st.success(f"✅ GTSRB Classification: Class {detection['class_id']} → {sign_meaning}")
                            elif model_type == "single_class":
                                st.error(f"❌ Single-class issue: Class {detection['class_id']} → {sign_meaning}")
                            else:
                                st.warning(f"⚠️ Generic Detection: Class {detection['class_id']} → {sign_meaning}")
                    else:
                        st.info("🤖 No classification available")
                    
                    # Confidence indicator with better styling
                    if detection['confidence'] > 0.8:
                        st.markdown('<div class="info-box">🟢 High confidence</div>', unsafe_allow_html=True)
                    elif detection['confidence'] > 0.5:
                        st.markdown('<div class="warning-box">🟡 Medium confidence</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">🔵 Low confidence</div>', unsafe_allow_html=True)
                
                with detail_col2:
                    st.markdown("**📝 Text Recognition:**")
                    
                    if detection['ocr_results']:
                        for j, ocr_result in enumerate(detection['ocr_results'][:3]):
                            confidence_icon = "🟢" if ocr_result['confidence'] > 0.7 else "🟡" if ocr_result['confidence'] > 0.5 else "🔴"
                            st.markdown(f"""
                            <div class="ocr-result">
                            {confidence_icon} <strong>"{ocr_result['text']}"</strong><br>
                            📊 OCR Confidence: {ocr_result['confidence']:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                        ℹ️ No readable text detected (likely symbolic/pictorial sign)
                        </div>
                        """, unsafe_allow_html=True)
    
    elif uploaded_file and not results['detections']:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ No traffic signs detected in this image</h4>
        <p><strong>Try these suggestions:</strong></p>
        <ul>
        <li>Lower the confidence threshold in the sidebar (try 0.05)</li>
        <li>Use an image with clearer, larger traffic signs</li>
        <li>Ensure the image contains actual traffic signs</li>
        <li>Try a different image with better lighting</li>
        <li>Check if your model is working correctly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo section for users without images
    if not uploaded_file:
        st.markdown("---")
        st.markdown("### 🖼️ Sample Images")
        st.markdown("Don't have a street image? Try downloading these samples:")
        
        col_sample1, col_sample2, col_sample3 = st.columns(3)
        
        with col_sample1:
            st.markdown("""
            <div class="info-box">
            <h4>🚫 Speed Limit Sign</h4>
            <p><a href="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Speed_limit_30_sign.svg/240px-Speed_limit_30_sign.svg.png" target="_blank">Download Sample</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sample2:
            st.markdown("""
            <div class="info-box">
            <h4>🛑 Stop Sign</h4>
            <p><a href="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Stop_sign_MUTCD.svg/240px-Stop_sign_MUTCD.svg.png" target="_blank">Download Sample</a></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sample3:
            st.markdown("""
            <div class="info-box">
            <h4>⚠️ Yield Sign</h4>
            <p><a href="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/MUTCD_R1-2.svg/240px-MUTCD_R1-2.svg.png" target="_blank">Download Sample</a></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with better styling
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;">
    <h4>🤖 <strong>Powered by:</strong> Streamlit • YOLOv8 • EasyOCR</h4>
    <p>🎓 <em>AI for Autonomous Driving Research Project</em></p>
    <p>⭐ <em>Built for educational and research purposes</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
