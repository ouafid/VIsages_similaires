from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import face_recognition
import os
import uvicorn

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

SIMILAR_IMAGES_FOLDER = './dataImg'

# Load the face signature database
signatures_db = np.load('FaceSignature_db2.npy', allow_pickle=True)
signatures_encodings = signatures_db[:, :-1].astype(np.float64)

def find_similar_faces(target_image, threshold=0.5):
    """Find similar faces in the database."""
    target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_encoding = face_recognition.face_encodings(target_image_rgb)
    
    if not target_encoding:
        return []
    
    target_encoding = target_encoding[0]
    distances = face_recognition.face_distance(signatures_encodings, target_encoding)
    similar_faces = [(signatures_db[i][-1], distances[i]) for i in range(len(distances)) if distances[i] <= threshold]
    return similar_faces

def get_similar_images_paths(similar_faces):
    """Get paths to similar images."""
    similar_images_paths = []
    for name, _ in similar_faces:
        image_path = os.path.join(SIMILAR_IMAGES_FOLDER, name + '.jpg')
        similar_images_paths.append(image_path)
    return similar_images_paths

@app.post("/faceID/")
async def Faces(uploaded_file: UploadFile = File(...)):
    file_bytes = await uploaded_file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"message": "Invalid image file"}
    
    # Debugging: Check image properties
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    similar_faces = find_similar_faces(img)
    similar_images_paths = get_similar_images_paths(similar_faces)
    return {"similar_faces": similar_images_paths}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
