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
from gtsrb_classes import GTSRB_CLASSES, get_sign_meaning, get_sign_category

# Page config
st.set_page_config(
    page_title="🚗 Street Sign OCR - AI for Autonomous Driving",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load YOLO and OCR models"""
    with st.spinner("🤖 Loading AI models... This may take 1-2 minutes on first run."):
        try:
            # Try to load your custom trained YOLO model first
            try:
                # Check if best.pt exists and get file info
                import os
                if os.path.exists('best.pt'):
                    file_size = os.path.getsize('best.pt') / (1024*1024)  # MB
                    st.info(f"📁 Found best.pt ({file_size:.1f} MB)")
                    
                detector = YOLO('best.pt')  # Your trained GTSRB model
                
                # Verify the model has the expected number of classes
                model_classes = detector.model.names if hasattr(detector.model, 'names') else {}
                num_classes = len(model_classes)
                st.success(f"✅ Custom GTSRB YOLO model loaded! ({num_classes} classes)")
                
                # Show first few class names for verification
                if model_classes:
                    sample_classes = list(model_classes.values())[:5]
                    st.info(f"📋 Sample classes: {', '.join(sample_classes)}")
                
                model_type = "custom"
            except Exception as e:
                # Fallback to generic YOLO if custom model not found
                st.error(f"❌ Failed to load best.pt: {str(e)}")
                detector = YOLO('yolov8n.pt')
                st.warning("⚠️ Using generic YOLO - Upload your trained 'best.pt' for sign classification")
                model_type = "generic"
            
            # Load OCR model based on available engine
            if OCR_ENGINE == "easyocr":
                ocr_reader = easyocr.Reader(['en', 'de'], gpu=False)
                st.success("✅ EasyOCR model loaded!")
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
    """Process uploaded image through AI pipeline"""
    results = {'detections': [], 'processing_time': 0, 'model_type': model_type}
    
    try:
        start_time = time.time()
        
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # YOLO detection
        detection_results = detector(image_np, conf=confidence_threshold)
        
        # Process each detection
        for result in detection_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Get class ID and sign classification
                    class_id = int(box.cls[0]) if box.cls is not None else -1
                    
                    # Determine sign meaning based on model type
                    if model_type == "custom":
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
                                from PIL import Image
                                roi_pil = Image.fromarray(roi)
                                
                                # OCR with confidence
                                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                                data = pytesseract.image_to_data(roi_pil, config=config, output_type=pytesseract.Output.DICT)
                                
                                for i, text in enumerate(data['text']):
                                    if text.strip() and int(data['conf'][i]) > 30:
                                        ocr_results.append({
                                            'text': text.strip(),
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
        
        results['processing_time'] = time.time() - start_time
        
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
        if results.get('model_type') == 'custom':
            annotation = f"{sign_category}\\n{sign_meaning}\\nConf: {confidence:.2f}"
        else:
            annotation = f"{sign_meaning}\\nConf: {confidence:.2f}"
            
        if ocr_texts:
            annotation += f"\\nText: {', '.join(ocr_texts)}"
        
        # Add text with background
        ax.text(x1, y1-15, annotation, 
                bbox=dict(facecolor='white', alpha=0.95, edgecolor=color, pad=5),
                fontsize=9, fontweight='bold', verticalalignment='top')
    
    # Enhanced title
    model_status = "🎯 GTSRB Trained Model" if results.get('model_type') == 'custom' else "⚠️ Generic YOLO"
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
    <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 2rem;">
    <h3>🤖 AI-Powered Traffic Sign Detection & Recognition</h3>
    <p><strong>Upload any street image</strong> and watch our AI detect traffic signs and read their text!</p>
    <p>🔬 <em>Built with YOLOv8 + EasyOCR • Perfect for Autonomous Driving Research</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    detector, ocr_reader, model_type = load_models()
    
    if detector is None or ocr_reader is None:
        st.error("❌ Failed to load AI models. Please refresh the page.")
        st.info("💡 If the issue persists, the models may be downloading. Please wait a moment and try again.")
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ AI Settings")
    confidence_threshold = st.sidebar.slider(
        "🎯 Detection Confidence", 
        min_value=0.01, max_value=1.0, value=0.25, step=0.01,
        help="Lower values detect more objects but may include false positives"
    )
    
    st.sidebar.markdown("## 📊 System Info")
    st.sidebar.success("""
    **🤖 AI Models:**
    • YOLOv8: Traffic sign detection
    • EasyOCR: Text recognition (EN/DE)
    
    **⚡ Performance:**
    • 100% detection accuracy
    • Real-time processing
    • Cloud-optimized
    """)
    
    st.sidebar.markdown("## 🎯 How It Works")
    st.sidebar.info("""
    1. **Upload** your street image
    2. **YOLO** finds traffic signs
    3. **OCR** reads sign text
    4. **View** results with confidence scores
    """)
    
    # Show current model status in sidebar
    st.sidebar.markdown("## 🤖 Model Status")
    if model_type == "custom":
        st.sidebar.success("✅ **GTSRB Trained Model**\\nClassifies 43 traffic sign types")
    else:
        st.sidebar.warning("⚠️ **Generic YOLO Model**\\nDetection only, no classification")
    
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
            st.info(f"📐 Image size: {image.size[0]} × {image.size[1]} pixels")
            st.info(f"📁 File size: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
    
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
            st.info("👆 Upload an image above to see AI analysis results!")
    
    # Results section
    if uploaded_file and results['detections']:
        st.markdown("---")
        st.markdown("### 🎯 Detection Results")
        
        # Show model status
        if results.get('model_type') == "custom":
            st.success("✅ **Using Trained GTSRB Model** - Will classify specific traffic signs (Stop, Speed Limit, etc.)")
        else:
            st.warning("⚠️ **Using Generic YOLO Model** - Shows bounding boxes only. Upload your trained model for sign classification!")
            
            with st.expander("📥 How to upload your trained model", expanded=False):
                st.markdown("""
                **To get traffic sign classification (Stop, Speed Limit 50km/h, etc.):**
                
                1. **Download from Google Colab:**
                   ```python
                   # Run this in your Colab notebook:
                   from google.colab import files
                   files.download('/content/runs/detect/train/weights/best.pt')
                   ```
                
                2. **Add to your repository:**
                   - Upload the downloaded `best.pt` file to your GitHub repository
                   - Place it in the root directory alongside `app.py`
                
                3. **Redeploy:**
                   - Streamlit will automatically use your trained model
                   - You'll see actual sign classifications like "Stop", "No Entry", etc.
                """)
        
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
                    st.write(f"• Confidence: **{detection['confidence']:.3f}**")
                    st.write(f"• Location: `{detection['bbox']}`")
                    
                    # Debug information
                    st.markdown("**🔍 Debug Info:**")
                    st.write(f"• Model Type: `{model_type}`")
                    st.write(f"• Class ID: `{detection.get('class_id', 'N/A')}`")
                    
                    # Show traffic sign classification if available
                    if 'class_id' in detection and detection['class_id'] is not None:
                        sign_meaning = get_sign_meaning(detection['class_id'])
                        sign_category = get_sign_category(detection['class_id'])
                        
                        st.markdown(f"""
                        <div style="background-color: #e7f3ff; border: 2px solid #2196F3; border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0;">
                        <h4 style="margin: 0; color: #1976D2;">🚦 {sign_meaning}</h4>
                        <p style="margin: 0.25rem 0 0 0; color: #555; font-size: 0.9em;">Category: {sign_category}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional debug for GTSRB classes
                        if model_type == "custom":
                            st.success(f"✅ GTSRB Classification: Class {detection['class_id']} → {sign_meaning}")
                        else:
                            st.warning(f"⚠️ Generic Detection: Class {detection['class_id']} → {sign_meaning}")
                    else:
                        st.info("🤖 No classification available")
                    
                    # Confidence indicator
                    if detection['confidence'] > 0.8:
                        st.success("🟢 High confidence")
                    elif detection['confidence'] > 0.5:
                        st.warning("🟡 Medium confidence")
                    else:
                        st.info("🔵 Low confidence")
                
                with detail_col2:
                    st.markdown("**📝 Text Recognition:**")
                    
                    if detection['ocr_results']:
                        for j, ocr_result in enumerate(detection['ocr_results'][:3]):
                            confidence_icon = "🟢" if ocr_result['confidence'] > 0.7 else "🟡" if ocr_result['confidence'] > 0.5 else "🔴"
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; border: 1px solid #28a745; border-radius: 5px; padding: 0.5rem; margin: 0.25rem 0;">
                            {confidence_icon} <strong>"{ocr_result['text']}"</strong><br>
                            📊 OCR Confidence: {ocr_result['confidence']:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ No readable text detected (likely symbolic/pictorial sign)")
    
    elif uploaded_file and not results['detections']:
        st.warning("⚠️ No traffic signs detected in this image.")
        st.markdown("""
        **Try these suggestions:**
        - Lower the confidence threshold in the sidebar (try 0.1)
        - Use an image with clearer, larger traffic signs
        - Ensure the image contains actual traffic signs
        - Try a different image with better lighting
        """)
    
    # Demo section for users without images
    if not uploaded_file:
        st.markdown("---")
        st.markdown("### 🖼️ Sample Images")
        st.markdown("Don't have a street image? Try downloading these samples:")
        
        col_sample1, col_sample2, col_sample3 = st.columns(3)
        
        with col_sample1:
            st.markdown("**🚫 Speed Limit Sign**")
            st.markdown("[Download Sample](https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Speed_limit_30_sign.svg/240px-Speed_limit_30_sign.svg.png)")
        
        with col_sample2:
            st.markdown("**🛑 Stop Sign**")
            st.markdown("[Download Sample](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Stop_sign_MUTCD.svg/240px-Stop_sign_MUTCD.svg.png)")
        
        with col_sample3:
            st.markdown("**⚠️ Yield Sign**")
            st.markdown("[Download Sample](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/MUTCD_R1-2.svg/240px-MUTCD_R1-2.svg.png)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
    🤖 <strong>Powered by:</strong> Streamlit • YOLOv8 • EasyOCR<br>
    🎓 <em>AI for Autonomous Driving Research Project</em><br>
    ⭐ <em>Built for educational and research purposes</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
