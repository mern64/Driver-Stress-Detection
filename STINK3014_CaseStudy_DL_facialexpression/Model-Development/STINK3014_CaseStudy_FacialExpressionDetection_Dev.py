import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Data Source
data = pd.read_csv("fer2013.csv")

# ==========================================
# NEW PART: Display 10 Random Records
# ==========================================
print("Displaying 10 random records...")

# Mapping just for display text
emotion_text = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# Your logic for Stress vs Calm
display_stress_classes = [0,1, 2, 4]

plt.figure(figsize=(14, 6))
# Pick 10 random indices
random_indices = random.sample(range(len(data)), 10)

for i, idx in enumerate(random_indices):
    row = data.iloc[idx]
    emotion_val = int(row['emotion'])

    # Determine label based on your logic
    if emotion_val in display_stress_classes:
        label_text = "Stress"
        color = 'red'
    else:
        label_text = "Calm"
        color = 'green'

    # Process image for display
    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    img = pixels.reshape(48, 48)

    # Plot
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{emotion_text.get(emotion_val)}\n({label_text})", color=color, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.show()
# ==========================================
# END OF NEW PART
# ==========================================

images = []
labels = []

# FER2013 mapping: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
stress_classes = [0,1,2, 4]  # Angry, Fear, Sad -> stress
calm_classes = [3, 5, 6]  # Happy, Surprise, Neutral -> calm

print("Processing data...")
for index, row in data.iterrows():
    emotion = int(row['emotion'])
    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    img = pixels.reshape(48, 48)

    if emotion in stress_classes:
        label = 1  # Stress
    else:
        label = 0  # Calm
    images.append(img)
    labels.append(label)

X = np.array(images).reshape(-1, 48, 48, 1) / 255.0
y = to_categorical(labels, num_classes=2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train CNN
# NOTE: See configuration suggestions below to improve this line
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2) # Change epoch = 30 , batch_size = 64

# Save trained model
model.save("stress_detector_cnn.h5")
print("Model saved as stress_detector_cnn.h5")