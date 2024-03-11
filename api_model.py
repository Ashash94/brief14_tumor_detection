from io import BytesIO
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import mlflow
from tensorflow.keras.models import load_model # Corrected import

app = FastAPI()

model = None

def normalize_images(X, target_size):
    # Your normalization logic here
    pass

@app.post('/predict/')
async def predict(img_bytes: UploadFile):
    img_bytes = BytesIO(await img_bytes.read())
    image_pil = Image.open(img_bytes)
    X_img = np.array(image_pil)[:, :, ::-1].astype('uint8')

    X_img = np.expand_dims(X_img, axis=0)

    X_img_norm = normalize_images(X_img,(224, 224))

    pred = model.predict(X_img_norm)

    return {'tumor_probability': float(pred[0][0])}

if __name__ == '__main__':
    model = load_model('my_model.h5') # Corrected model loading
    uvicorn.run(app)
# Que cherchons nous a faire 
# modifier add patient de sorte a se que l'on puisse charger une image afin qu'il puisse faire appel a l'API et faire la prediction et sauver le patient avec sont scan et avec la prediction html 