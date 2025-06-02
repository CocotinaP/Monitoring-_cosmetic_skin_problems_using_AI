# IMPROVED FACE PROBLEM DETECTION MODEL

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# 1. IMPROVED MODEL ARCHITECTURE (Transfer Learning)
def create_improved_model(num_classes=4):
    """
    Advanced model using ResNet50V2 with transfer learning
    Much better than the simple CNN
    """
    base_model = tf.keras.applications.ResNet50V2(
        weights='imagenet',  # Use pre-trained weights
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers (transfer learning)
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 2. FACE DETECTION & CROPPING
def detect_and_crop_face(image_path):
    """
    Automatically detect and crop face from image
    This improves accuracy by focusing on the face area
    """
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100)
    )
    
    if len(faces) > 0:
        # Take the largest face (most likely the main subject)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face
        
        # Add some padding around the face
        padding = int(0.2 * min(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Crop face region
        face_img = img[y:y+h, x:x+w]
        
        # Resize to model input size
        face_img = cv2.resize(face_img, (224, 224))
        return face_img
    else:
        print("No face detected, using full image")
        return cv2.resize(img, (224, 224))
    
  # 3. ADVANCED DATA PREPROCESSING
def preprocess_image_advanced(image_path):
    """
    Advanced preprocessing with face detection and enhancement
    """
    # Detect and crop face
    face_img = detect_and_crop_face(image_path)
    
    # Convert BGR to RGB
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Enhance image quality
    # Histogram equalization for better contrast
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize pixel values
    enhanced_img = enhanced_img.astype(np.float32) / 255.0
    
    # Add batch dimension
    enhanced_img = np.expand_dims(enhanced_img, axis=0)
    
    return enhanced_img

# 4. CONFIDENCE-BASED PREDICTION
def predict_with_confidence(model, image_path, class_names, confidence_threshold=0.6):
    """
    Make predictions with confidence scoring
    Only return predictions above threshold
    """
    # Preprocess image
    processed_img = preprocess_image_advanced(image_path)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    max_confidence = np.max(predictions[0])
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    
    # Check confidence threshold
    if max_confidence < confidence_threshold:
        return {
            'prediction': 'uncertain',
            'confidence': max_confidence,
            'message': f'Low confidence ({max_confidence:.2%}). Image may be unclear or show multiple conditions.',
            'all_probabilities': {class_names[i]: predictions[0][i] for i in range(len(class_names))}
        }
    else:
        return {
            'prediction': predicted_class,
            'confidence': max_confidence,
            'message': f'Detected {predicted_class} with {max_confidence:.2%} confidence',
            'all_probabilities': {class_names[i]: predictions[0][i] for i in range(len(class_names))}
        }
    
# 5. MULTI-CONDITION DETECTION (if you want to detect multiple problems at once)
def create_multilabel_model(num_classes=4):
    """
    Model that can detect multiple skin conditions simultaneously
    """
    base_model = tf.keras.applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Binary crossentropy for multi-label
        metrics=['binary_accuracy']
    )
    
    return model

# 6. FIXED TRAINING WITH DATA AUGMENTATION
def create_advanced_data_generators(data_dir, batch_size=32):
    """
    Create data generators with advanced augmentation (FIXED)
    """
    # FIXED: Removed problematic preprocessing function
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        validation_split=0.2,
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation, only rescaling)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',  # Changed to sparse for sparse_categorical_crossentropy
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',  # Changed to sparse for sparse_categorical_crossentropy
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator


# 7. USAGE EXAMPLE 
def improved_prediction_function(model, class_names):
    """
    Enhanced version of your current prediction function with multiple image analysis
    FIXED: Proper tkinter window management for macOS
    """
    import tkinter as tk
    from tkinter import filedialog
    
    print("🔬 SKIN ANALYSIS APP - Multiple Image Analyzer")
    print("="*60)
    
    # Create root window once and reuse it
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Fix dialog positioning on macOS
    root.geometry("1x1+400+300")  # Small window positioned in center-ish area
    root.update_idletasks()  # Force geometry update
    
    try:
        while True:
            print("\n📷 Select an image to analyze (or cancel to exit)")
            
            # Ensure proper window positioning before dialog
            root.lift()  # Bring to front
            root.attributes('-topmost', True)  # Keep on top temporarily
            
            # Use the existing root window for file dialog
            image_path = filedialog.askopenfilename(
                title="Select an image for skin analysis",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
                parent=root
            )
            
            # Reset topmost attribute
            root.attributes('-topmost', False)
            
            if not image_path:
                print("👋 No image selected. Exiting analysis app.")
                break
            
            print(f"\n🔍 Analyzing image: {os.path.basename(image_path)}")
            print("-" * 50)
            
            try:
                # Make prediction with confidence
                result = predict_with_confidence(model, image_path, class_names)
                
                print(f"\n{'='*50}")
                print("🎯 SKIN ANALYSIS RESULTS")
                print(f"{'='*50}")
                print(f"📊 Prediction: {result['prediction'].upper()}")
                print(f"🎯 Confidence: {result['confidence']:.2%}")
                print(f"💬 Message: {result['message']}")
                
                print(f"\n📈 All Probabilities:")
                for condition, prob in result['all_probabilities'].items():
                    bar_length = int(prob * 20)  # Create a simple progress bar
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"  {condition:12} {bar} {prob:.2%}")
                
                # Get product recommendations if confident
                if result['prediction'] != 'uncertain':
                    print(f"\n🧴 PRODUCT RECOMMENDATIONS for {result['prediction'].upper()}:")
                    print("=" * 50)
                    try:
                        # Import the function from your notebook or local file
                        from local_image_predictor import get_product_recommendations_from_excel
                        recommendations = get_product_recommendations_from_excel(result['prediction'])
                        for rec in recommendations:
                            print(rec)
                    except ImportError:
                        print("📦 Product recommendation function not available.")
                        print("   Add the Excel file and functions to your environment for recommendations.")
                else:
                    print(f"\n⚠️  LOW CONFIDENCE RESULT")
                    print("Try with a clearer image or different lighting for better results.")
                    
            except Exception as e:
                print(f"❌ Error analyzing image: {e}")
            
            # Ask if user wants to analyze another image
            print(f"\n{'='*60}")
            while True:
                choice = input("🤔 Would you like to analyze another image? (y/n): ").lower().strip()
                if choice in ['y', 'yes', '']:
                    break
                elif choice in ['n', 'no']:
                    print("👋 Thank you for using the Skin Analysis App!")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
    
    finally:
        # Properly cleanup the root window
        try:
            root.quit()
            root.destroy()
        except:
            pass  # Ignore any cleanup errors

# Alternative: Simple batch analysis function
def batch_analysis_function(model, class_names):
    """
    Analyze multiple images in batch mode
    FIXED: Proper tkinter window management
    """
    import tkinter as tk
    from tkinter import filedialog
    
    print("📁 BATCH SKIN ANALYSIS")
    print("="*50)
    
    # Create root window once
    root = tk.Tk()
    root.withdraw()
    
    # Fix dialog positioning on macOS
    root.geometry("1x1+400+300")  # Small window positioned in center-ish area
    root.update_idletasks()  # Force geometry update
    
    try:
        # Ensure proper window positioning before dialog
        root.lift()  # Bring to front
        root.attributes('-topmost', True)  # Keep on top temporarily
        
        # Select multiple files
        image_paths = filedialog.askopenfilenames(
            title="Select multiple images for batch analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
            parent=root
        )
        
        # Reset topmost attribute
        root.attributes('-topmost', False)
        
        if not image_paths:
            print("No images selected.")
            return
        
        print(f"Selected {len(image_paths)} images for analysis...")
        
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Analyzing: {os.path.basename(image_path)}")
            
            try:
                result = predict_with_confidence(model, image_path, class_names)
                result['filename'] = os.path.basename(image_path)
                results.append(result)
                
                print(f"  Result: {result['prediction']} ({result['confidence']:.1%})")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # Summary report
        print(f"\n{'='*60}")
        print("📊 BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        for result in results:
            print(f"{result['filename']:25} → {result['prediction']:12} ({result['confidence']:.1%})")
    
    finally:
        # Properly cleanup the root window
        try:
            root.quit()
            root.destroy()
        except:
            pass

# Quick single image analysis function
def quick_analysis_function(model, class_names, image_path):
    """
    Quick analysis for a specific image path
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print(f"🔍 Quick analysis: {os.path.basename(image_path)}")
    
    try:
        result = predict_with_confidence(model, image_path, class_names)
        print(f"Result: {result['prediction']} ({result['confidence']:.1%})")
        print(f"Message: {result['message']}")
        
        if result['prediction'] != 'uncertain':
            try:
                from local_image_predictor import get_product_recommendations_from_excel
                recommendations = get_product_recommendations_from_excel(result['prediction'])
                print(f"\nTop 3 Recommendations:")
                for rec in recommendations[:3]:
                    print(f"  {rec}")
            except ImportError:
                pass
                
    except Exception as e:
        print(f"Error: {e}")

# 8. SIMPLE VERSION FOR TESTING
def create_simple_improved_model(num_classes=4):
    """
    Simpler version for testing if the complex model has issues
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    class_names = ['acnes', 'blackheads', 'darkspots', 'wrinkles']  # Fixed to match your dataset
    
    print("Creating improved model...")
    try:
        model = create_improved_model(len(class_names))
        print("✓ Model created successfully")
    except Exception as e:
        print(f"Error creating advanced model: {e}")
        print("Falling back to simple improved model...")
        model = create_simple_improved_model(len(class_names))
    
    # COMMENTED OUT TRAINING - SKIPPING TO PREDICTION
    print("Creating data generators...")
    try:
        train_gen, val_gen = create_advanced_data_generators("dataset/SkinMate-Dataset")
        print("✓ Data generators created successfully")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Classes found: {train_gen.class_indices}")
    except Exception as e:
        print(f"Error creating data generators: {e}")
        return
    
    print("Starting training...")
    try:
        history = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=10,  # Reduced epochs for testing
            verbose=1
        )
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    print("⚠️  Note: Using untrained model for demonstration purposes.")
    print("   For actual predictions, you should train the model first.")
    print("   Uncomment the training section above to train the model.")
    
    # ANALYSIS MODE MENU
    print(f"\n{'='*60}")
    print("🎯 SKIN ANALYSIS APP - Choose Analysis Mode")
    print(f"{'='*60}")
    print("1. 📷 Interactive Mode (analyze multiple images one by one)")
    print("2. 📁 Batch Mode (analyze multiple images at once)")
    print("3. 🔍 Quick Mode (analyze specific image path)")
    print("4. ❌ Exit")
    
    while True:
        choice = input("\nSelect mode (1-4): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting Interactive Analysis Mode...")
            improved_prediction_function(model, class_names)
            break
        elif choice == '2':
            print("\n🚀 Starting Batch Analysis Mode...")
            batch_analysis_function(model, class_names)
            break
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            print("\n🚀 Starting Quick Analysis...")
            quick_analysis_function(model, class_names, image_path)
            break
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()

