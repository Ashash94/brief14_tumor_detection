from io import BytesIO
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = FastAPI()

# Load your model
model = load_model('my_model.h5')

def normalize_images(X, target_size):
    normalized_images = [None] * len(X)

    for i, img in enumerate(X):
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img

        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cropped_img = img[y:y+h, x:x+w]
            normalized_images[i] = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        else:
            normalized_images[i] = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return np.array(normalized_images)


@app.post('/predict/')
async def predict(img_bytes: UploadFile):

    img_bytes = BytesIO(await img_bytes.read())
    image_pil = Image.open(img_bytes)
    X_img = np.array(image_pil)[:, :, ::-1].astype('uint8')

    X_img = np.expand_dims(X_img, axis=0)
    X_img_norm = normalize_images(X_img, (224, 224))

    pred = model.predict(X_img_norm)
    tumor_probability = float(pred[0][0])

    return {'tumor_probability': tumor_probability}


if __name__ == '__main__':
    uvicorn.run(app, port=8001)
