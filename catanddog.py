# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define data paths (replace with your actual paths)
train_data_dir = "path/to/your/train/data"
test_data_dir = "path/to/your/test/data"

# Set image dimensions
img_width, img_height = 150, 150

# Create data generators with image augmentation for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# Define the model architecture (replace with more complex architectures if needed)
model = tf.keras.Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model (replace with desired evaluation metrics)
loss, accuracy = model.evaluate(validation_generator)
print("Accuracy:", accuracy)

# Predict on a new image (replace with your image path)
new_image = tf.keras.preprocessing.image.load_img("path/to/your/image.jpg", target_size=(img_width, img_height))
new_image_array = tf.keras.preprocessing.image.img_to_array(new_image)
new_image_batch = new_image_array / 255.0
new_image_batch = np.expand_dims(new_image_batch, axis=0)
prediction = model.predict(new_image_batch)

if prediction[0][0] > 0.5:
  print("Predicted: Cat")
else:
  print("Predicted: Dog")