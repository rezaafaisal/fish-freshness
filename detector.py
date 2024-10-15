from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageDraw


class Detector:

   def __init__(self):
       self.model = YOLO("yolov8n.pt")

   def detect(self, image_path):
      image = Image.open("/content/image1.jpg")
      output = self.model(image)
      results = Detections.from_ultralytics(output[0])
   
   def draw(self, image_path):
      # draw box on image
      draw = ImageDraw.Draw(image:)

      # Loop through detected objects and draw bounding boxes
      for det in results.xyxy[0]:
         # det format: [x_min, y_min, x_max, y_max, confidence, class]
         x_min, y_min, x_max, y_max, = results.xyxy[0]

         # Draw bounding box
         draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

         # Annotate with class name and confidence
         draw.text((x_min, y_min), text='test', fill="red")

      # Display or save the image with bounding boxes
      image.show()

       