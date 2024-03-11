import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from PIL import Image
from io import BytesIO
import keras 
from keras.models import load_model
from pydantic import BaseModel


app = FastAPI()
# Connexion à la base de données MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["braintumor"] 

# load model
loaded_model = load_model('my_model.h5')
model = load_model('my_model.h5')
# Modèle Pydantic pour les données du patient
class PatientModel(BaseModel):
    name: str
    age: int
    gender: str


# Modèles Pydantic pour la modification du patient
class PatientUpdateModel(BaseModel):
    name: str
    age: int
    gender: str


# Modèles Pydantic pour la visualisation des patients
class PatientViewModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str


# Modèle Pydantic pour les prédictions (à adapter selon vos besoins)
class PredictionModel(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    pass


# Montez le répertoire 'static' pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")


# Route pour ajouter un patient
@app.get("/add_patient", response_class=HTMLResponse)
def add_patient(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})


from fastapi import UploadFile

@app.post("/add_patient")
async def add_patient_post(patient: PatientModel, img_bytes: UploadFile = File(...)):
    # Process the image and make a prediction
    img_bytes = BytesIO(await img_bytes.read())
    image_pil = Image.open(img_bytes)
    X_img = np.array(image_pil)[:, :, ::-1].astype('uint8')
    X_img = np.expand_dims(X_img, axis=0)
    X_img_norm = normalize_images(X_img, (224, 224))
    pred = model.predict(X_img_norm)
    
    # Convert prediction to a suitable format
    tumor_probability = float(pred[0][0])
    
    # Prepare patient data including the prediction
    patient_data = patient.dict()
    patient_data["scan"] = img_bytes.getvalue().decode('utf-8') # Assuming you want to store the image as a string
    patient_data["prediction"] = tumor_probability
    
    # Insert the patient data into the database
    db.patients.insert_one(patient_data)
    
    return JSONResponse(content={"message": "Patient added successfully", "patient_id": str(patient_data["_id"])})


# Route pour visualiser tous les patients
@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(request: Request):
    # Récupérer tous les patients depuis la base de données
    patients = [PatientViewModel(id=str(patient['_id']), **patient) for patient in db.patients.find()]
    return templates.TemplateResponse("view_patients.html", {"request": request, "patients": patients})


# Route pour éditer un patient
@app.get("/edit_patient/{patient_id}", response_class=HTMLResponse)
async def edit_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
    return templates.TemplateResponse("edit_patient.html", {"request": request, "patient": patient,
                                                            "patient_id": patient_id})


@app.post("/edit_patient/{patient_id}")
async def edit_patient_post(patient_id: str, patient: PatientUpdateModel):
    # Mettre à jour le patient dans la base de données
    db.patients.update_one({"_id": ObjectId(patient_id)}, {"$set": patient.model_dump()})
    return RedirectResponse(url="/view_patients")

if __name__ == '__main__':
    uvicorn.run(app)