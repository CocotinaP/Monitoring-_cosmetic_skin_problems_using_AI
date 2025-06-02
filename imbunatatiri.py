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

