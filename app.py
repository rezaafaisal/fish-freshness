from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from classifier import Classifier
from detector import Detector
from utils import save_image

app = FastAPI()

app.mount('/images', StaticFiles(directory="images"), name="images")

classifier = Classifier()
detector = Detector()

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

@app.get("/")
def home():
   return "Hello, World!"

@app.post("/predict")
async def predict(image: UploadFile = File(...)):

   (image_path, _) = save_image(image, "original")
   
   detection = detector.detect(image_path)
   (prediction, prob, acc) = classifier.predict(image_path)
   print("detection", detection)
   body = {
      "prediction": prediction,
      "prob": prob,
      "accuracy": acc
   }
   return JSONResponse(body)

@app.get("/tes")
def tes(image_name: str, request: Request):


   
   full_image_url = request.url_for('images', path=image_name)

   return {"image_url": full_image_url}