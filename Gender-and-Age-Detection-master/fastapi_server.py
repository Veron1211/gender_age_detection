from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import uvicorn
from typing import List, Dict, Union

app = FastAPI()

# Load OpenCV Pre-trained Models
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
# Corrected face detection model loading
face_net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def highlight_face(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            # Ensure coordinates are within frame boundaries
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width-1, x2), min(frame_height-1, y2)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, face_boxes

@app.post("/detect_gender_age/")
async def detect_gender_age(image: UploadFile = File(...)) -> Dict[str, Union[str, List[Dict[str, Union[str, List[int]]]]]]:
    # Read Image
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Failed to decode image"}

    # Detect faces
    img, face_boxes = highlight_face(face_net, img)
    print(f"Detected face boxes: {face_boxes}")

    if not face_boxes:
        return {"error": "No face detected"}

    # Process first face
    results = []
    for faceBox in face_boxes:
        x1, y1, x2, y2 = faceBox
        face = img[y1:y2, x1:x2]
        padding = 20

        # Check if face region is valid
        if face.size == 0:
            continue

        # Apply padding
        face = img[max(0, y1 - padding): min(y2 + padding, img.shape[0] - 1),
                   max(0, x1 - padding): min(x2 + padding, img.shape[1] - 1)]

        # Predict Gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        print(f'Gender: {gender}')

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # Append result for this face
        results.append({"gender": gender, "age": age, "faceBox": faceBox})

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
