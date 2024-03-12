from io import BytesIO
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse

app = FastAPI()

# Define your Pydantic model for patient data
class PatientModel(BaseModel):
    name: str
    birthdate: int
    gender: str
    # Add other fields as necessary

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

def decode_image(encoded_image: str, output_path: str):
    decoded_image = base64.b64decode(encoded_image)
    with open(output_path, "wb") as image_file:
        image_file.write(decoded_image)

@app.post('/predict/')
async def predict(patient: PatientModel, encoded_image: str):
    try:
        # Decode the base64 image
        decoded_image = base64.b64decode(encoded_image)
        img_bytes = BytesIO(decoded_image)
        image_pil = Image.open(img_bytes)
        X_img = np.array(image_pil)[:, :, ::-1].astype('uint8')

        X_img = np.expand_dims(X_img, axis=0)
        X_img_norm = normalize_images(X_img, (224, 224))

        pred = model.predict(X_img_norm)
        tumor_probability = float(pred[0][0])

        patient_data = patient.dict()
        patient_data["prediction"] = tumor_probability

        # Insert the patient data into the database
        # db.patients.insert_one(patient_data)

        return JSONResponse(content={"message": "Patient added successfully", "patient_id": str(patient_data["_id"]), "prediction": tumor_probability})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, port=8001)
