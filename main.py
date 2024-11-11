import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Prepare Data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'dataset/',  # Path to the dataset
    target_size=(224, 224), #resize all images to 224x224
    batch_size=32, #batch size, adjust if needed
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Store class indices for later use
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping of indices to class names

# Load the pre-trained MobileNetV2 model and add custom layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

#Setting up the neural network with pretrained base model using Keras API
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_indices), activation='softmax')  # Number of classes in your dataset
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model (uncomment to train)
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
model.save('traffic_signal_model.h5')

# Load the trained model
model = tf.keras.models.load_model('traffic_signal_model.h5')

# Add these constants
CONFIDENCE_THRESHOLD = 0.70  # Adjust this value based on your needs
MIN_SIGN_AREA = 1000  # Minimum area for a detected contour to be considered a sign

def detect_potential_signs(frame):
    """
    Detect potential traffic signs using basic computer vision techniques
    Returns a boolean indicating if a potential sign was detected
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for common traffic sign colors (red, blue)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([130, 255, 255])
    
    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour meets the minimum area requirement
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_SIGN_AREA:
            return True, contour
    
    return False, None

def preprocess_frame(frame, target_size=(224, 224)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, target_size)
    frame_array = np.expand_dims(frame_resized, axis=0)
    return preprocess_input(frame_array)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the trained model
model = tf.keras.models.load_model('traffic_signal_model.h5')

# Run real-time detection
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Frame capture failed or returned None.")
        break
    
    # First, check if there's a potential sign in the frame
    sign_detected, contour = detect_potential_signs(frame)
    
    if sign_detected:
        # Preprocess the frame
        input_frame = preprocess_frame(frame)
        
        # Make predictions
        predictions = model.predict(input_frame, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Only show prediction if confidence is above threshold
        if confidence > CONFIDENCE_THRESHOLD:
            predicted_label = class_labels[predicted_class_idx]
            text = f"{predicted_label}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw rectangle around the detected sign if contour exists
            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No sign detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No sign detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Traffic Signal Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()