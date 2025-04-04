from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import uvicorn
import asyncio
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Load models once at startup
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
face_net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
executor = ThreadPoolExecutor(max_workers=4)

PADDING = 20  # padding added around the detected face box

def detect_faces(frame: np.ndarray) -> List[List[int]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Add padding and clip within frame bounds
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w - 1, x2 + PADDING)
            y2 = min(h - 1, y2 + PADDING)

            faces.append([x1, y1, x2, y2])

    # If multiple faces are detected, select the largest one
    if len(faces) > 1:
        largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))  # Area of the bounding box
        faces = [largest_face]

    return faces


def process_face(face: np.ndarray) -> Dict[str, str]:
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    
    return {
        "gender": gender,
        "age": age,
    }

@app.post("/detect/")
async def detect(image: UploadFile = File(...)) -> Dict[str, Union[str, List[Dict]]]:
    loop = asyncio.get_running_loop()
    
    # Read and decode image
    data = await image.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}

    # Detect faces
    faces = await loop.run_in_executor(executor, detect_faces, frame)
    if not faces:
        return {"results": []}

    # Process the largest face
    processed = []
    for x1, y1, x2, y2 in faces:
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        result = await loop.run_in_executor(executor, process_face, face_img)
        result.update({
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })
        processed.append(result)

    return {"results": processed}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)