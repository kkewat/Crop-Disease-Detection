import os
import cv2
import numpy as np

print("preprocessing image")
# Adding the dataset address
data_path = 'F:\\DataSets\\MY_CROP_DISEASE_DATASET'

"""**Defining The Function to Convert Images from RGB to HSV**"""

def preprocess_image(image, target_size=(224,224)):
    # If image is a file path, read it
    if isinstance(image, str):
        img = cv2.imread(image)  # Reading the image
    else:
        img = image  # Assuming image is already read

    # Check if image is read successfully
    if img is None:
        raise ValueError("Unable to read image")

    # Converting the Color of the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Resizing the image
    img = cv2.resize(img, target_size)
    return img

# List all classes in the dataset
classes = os.listdir(data_path)
print(classes)

# Initialize lists to store images and labels
images = []
labels = []

# Looping through each classes
for class_name in classes:
    class_path = os.path.join(data_path, class_name)

    # Looping through each image
    for image_name in os.listdir(class_path):
        try:
            print(image_name)
            print(class_path)
            image_path = os.path.join(class_path, image_name)
            # Preprocess the image and append it in the list
            img = preprocess_image(image_path)
            images.append(img)
            labels.append(class_name)
        except Exception as e:
            # Print an error message and continue to the next iteration
            print(f"Error processing {image_name}: {e}")
            continue

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)


# Save images and labels to a more efficient format
np.save('images.npy', images)
np.save('labels.npy', labels)
