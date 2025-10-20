import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
import os

# --- Paths ---
image_dir = r"E:\ML Practice\Starting over\dress_recommender\Fashion_Product_Images_(Small)\images"

# --- Load filenames and embeddings ---
filenames = pickle.load(open('filename.pkl','rb'))
feature_list = pickle.load(open('embeddings.pkl','rb'))
feature_list = np.array(feature_list)  # Convert to 2D array

# --- Load model ---
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# --- Load query image ---
img = image.load_img(r"D:/S6d67f94bdbd04140b7e0327b5f25636bX.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# --- Nearest Neighbors ---
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized_result])
print(indices)

# --- Display top 5 similar images ---
for file_idx in indices[0]:
    full_path = os.path.join(image_dir, filenames[file_idx])
    temp_img = cv2.imread(full_path)
    if temp_img is not None:
        cv2.imshow('output', cv2.resize(temp_img,(512,512)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Could not read {full_path}")
