from fastapi import UploadFile
from PIL import Image


def save_image(file: UploadFile, filename: str):
   try:
      image_path = f"images/{filename}.jpg"
      image = Image.open(file.file)
      if image.mode in ("RGBA", "P"): 
         image = image.convert("RGB")
      image.save(image_path, "JPEG")
      return (image_path, image)
   except Exception as e:
      print(e)
      raise e
   
def load_image(filename: str) -> Image:
   return Image.open(f"images/{filename}.jpg")

