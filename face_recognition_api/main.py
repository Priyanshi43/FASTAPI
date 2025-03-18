from fastapi import FastAPI, UploadFile, File
import face_recognition
import os
import pickle
import shutil

app = FastAPI()

dataset_path = "app/static/image-dataset"
known_faces_path = "app/models/knownfaces.pkl"

# Ensure dataset and model folder exist
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(os.path.dirname(known_faces_path), exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(dataset_path, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "filename": file.filename}

@app.post("/train/")
def train_faces():
    known_encodings = []
    known_names = []

    for file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(file.split(".")[0])
    
    with open(known_faces_path, "wb") as f:
        pickle.dump((known_encodings, known_names), f)
    
    return {"message": "Face Encodings Training Done!", "total_faces": len(known_encodings)}

@app.get("/status/")
def check_status():
    return {"message": "Face Recognition API is running"}
