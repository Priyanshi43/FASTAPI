from fastapi import FastAPI, UploadFile, File
import cv2
import face_recognition
import pickle
import os
import time
import shutil
from fastapi.responses import FileResponse

app = FastAPI()

known_faces_path = "app/models/knownfaces.pkl"
recognized_images_folder = "app/static/recognized-images"
os.makedirs(recognized_images_folder, exist_ok=True)

# Load trained faces
def load_known_faces():
    if not os.path.exists(known_faces_path):
        return None, None
    with open(known_faces_path, "rb") as f:
        return pickle.load(f)

@app.post("/recognize/")
async def recognize_faces(file: UploadFile = File(...)):
    known_face_encodings, known_face_names = load_known_faces()
    if known_face_encodings is None:
        return {"error": "No trained data found. Please train first."}
    
    file_path = os.path.join(recognized_images_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    test_image = face_recognition.load_image_file(file_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        threshold = 0.45
        best_match_index = None
        min_distance = float("inf")

        for i, distance in enumerate(face_distances):
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match_index = i

        name = "Unknown"
        if best_match_index is not None:
            name = known_face_names[best_match_index]

        cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_image_path = os.path.join(recognized_images_folder, f"recognized_{timestamp}.jpeg")
    cv2.imwrite(output_image_path, test_image)
    
    return {"message": "Recognition complete", "saved_image": output_image_path}

@app.get("/download/{image_name}")
def download_image(image_name: str):
    image_path = os.path.join(recognized_images_folder, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}

@app.get("/status/")
def check_status():
    return {"message": "Face Recognition API is running"}
