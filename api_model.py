from io import BytesIO
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import mlflow
from tensorflow.keras.models import load_model # Corrected import
from io import BytesIO

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import mlflow

app = FastAPI()

model = None


def normalize_images(X, target_size):
    normalized_images = [None] * len(X)

    for i, img in enumerate(X):
        if len(img.shape) == 3:
            # Convertir en niveaux de gris si c'est pas déjà le cas
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img

        # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
        denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Détecter les contours pour trouver le crop optimal
        _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Trouver le contour avec la plus grande aire
            max_contour = max(contours, key=cv2.contourArea)

            # Obtenir les coordonnées du rectangle englobant
            x, y, w, h = cv2.boundingRect(max_contour)

            # Cropper l'image pour obtenir la région d'intérêt
            cropped_img = img[y:y+h, x:x+w]

            # Redimensionner à target_size (pour s'assurer que toutes les images ont la même taille)
            normalized_images[i] = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        else:
            # Redimensionner à target_size si aucun contour n'est détecté
            normalized_images[i] = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return np.array(normalized_images)


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
    model = load_model('my_model.h5') 
    uvicorn.run(app, port=8001)
# Que cherchons nous a faire 
# modifier add patient de sorte a se que l'on puisse charger une image afin qu'il puisse faire appel a l'API et faire la prediction et sauver le patient avec sont scan et avec la prediction html  