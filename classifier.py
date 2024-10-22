import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class Classifier:
   def __init__(self):
      self.model = self.load_model()

   def load_model(self):
      model_path = '/home/anonymous/fish/models/inception_resnet_v2.keras'
      model = tf.keras.models.load_model(model_path)
      return model
   
   def predict(self, image_path):
      image_height = 224
      image_width = 224

      img = image.load_img(image_path, target_size=(image_height, image_width))

      # Convert the image to a numpy array
      img_array = image.img_to_array(img)

      # Step 3: Preprocess the image (rescale, reshape)
      img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
      img_array = img_array / 255.0  # Rescale pixel values

      # Step 4: Make the prediction
      y_prob = self.model.predict(img_array)

      # Step 5: Convert prediction to class (0 -> fresh, 1 -> not_fresh)
      y_pred = (y_prob > 0.5).astype(int)

      # Step 6: Map the prediction to the class name
      class_names = {0: 'fresh', 1: 'not_fresh'}
      predicted_class = class_names[y_pred[0][0]]

      # Display the result
      print(y_prob)
      print(f"Prediction: {predicted_class}")

      probs = y_prob[0][0]*100

      accuracy = probs/50*100 if probs <= 50 else (probs-50)/50*100
      
      return (predicted_class, probs, accuracy)
   