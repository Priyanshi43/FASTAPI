﻿# FASTAPI

Following Steps:-

*FASTAPI FaceDetected Documentation*

-FastAPI automatically generates interactive API documentation using Swagger UI and ReDoc. You can access them at:
Swagger UI: http://127.0.0.1:8000/docs
ReDoc: http://127.0.0.1:8000/redoc.

If you haven't already, run your FastAPI app using:
      uvicorn main:app --reload6

-Customize Documentation with OpenAPI Tags
Use tags to categorize endpoints:
 @app.post("/detect_faces/", tags=["Face Detection"])
This groups similar endpoints under "Face Detection" in the Swagger UI.


-Run and Access Documentation
Start the FastAPI server:
uvicorn Fastapi_facedetect:app --reload
Open:
Swagger UI: http://127.0.0.1:8000/docs.
ReDoc: http://127.0.0.1:8000/redoc.


