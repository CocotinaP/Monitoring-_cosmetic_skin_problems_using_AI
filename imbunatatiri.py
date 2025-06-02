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