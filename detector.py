from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw

import os


class Detector:

   def __init__(self):
       self.model = YOLO("models/yolo8.pt")

   def detect(self, image_path):
      try:
         image = Image.open(image_path)
         output = self.model(image)
         results = Detections.from_ultralytics(output[0])

         self.crop(image, results)
         self.draw(image, results)
      except:
         raise
      
   def crop(self, image, bounding_box, filename="cropped"):
      try:
         x_min, y_min, x_max, y_max, = bounding_box.xyxy[0]

         img_cropped = image.crop((x_min, y_min, x_max, y_max))
         img_cropped.save(f"images/{filename}.jpg")
         pass

      except:
         raise
   
   def draw(self, image: Image, bounding_box, filename="drawn"):
      # draw box on image

      try:
         drawn = ImageDraw.Draw(image)

         # Loop through detected objects and draw bounding boxes
         for det in bounding_box.xyxy[0]:
            # det format: [x_min, y_min, x_max, y_max, confidence, class]
            x_min, y_min, x_max, y_max, = bounding_box.xyxy[0]

            # Draw bounding box
            drawn.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

            # Annotate with class name and confidence
            drawn.text((x_min, y_min), text='Mata Ikan', fill="red")
         
         image.save(f"images/{filename}.jpg")
      except Exception as e:
         raise e

       