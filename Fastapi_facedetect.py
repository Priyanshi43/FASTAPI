from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from starlette.responses import Response

app = FastAPI()

# Initialize MTCNN detector
try:
    detector = MTCNN()
except Exception as e:
    raise RuntimeError(f"Error loading MTCNN: {e}")

@app.get("/", summary="Check API Status", response_description="Returns a success message")
async def root():
    """Returns a message confirming the API is running."""
    return {"message": "API is running successfully!"}

@app.post("/detect_faces/", summary="Detect Faces in an Image")
async def detect_faces(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        
        # Convert image bytes to numpy array
        image_np = np.frombuffer(contents, np.uint8)
        if image_np.size == 0:
            raise HTTPException(status_code=400, detail="Empty or corrupted image file")
        
        # Decode the image
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format or corrupted file")
        
        # Convert BGR to RGB for MTCNN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector.detect_faces(image_rgb)
        if not faces:
            return {"message": "No faces detected in the image"}

        # Draw bounding boxes around detected faces
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)  # Ensure values are non-negative
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Encode image back to bytes
        _, encoded_img = cv2.imencode('.jpg', image)
        if not encoded_img:
            raise HTTPException(status_code=500, detail="Error encoding image")

        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
