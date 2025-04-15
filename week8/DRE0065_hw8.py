import cv2
import os
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern

EAR_THRESHOLD, LBP_THRESHOLD, FRAME_WINDOW = 0.20, 7.0, 5

def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_lbp_mean(roi_gray):
    lbp = local_binary_pattern(roi_gray, P=8, R=1, method="uniform")
    return lbp.mean()

def load_annotations(filepath):
    intervals = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2: intervals.append((int(parts[0]), int(parts[1])))
    return intervals

def is_closed_ground_truth(frame_idx, intervals):
    for start, end in intervals:
        if start <= frame_idx <= end: return True
    return False

face_cascade = cv2.CascadeClassifier("landmark_models/lbpcascade_frontalface_improved.xml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("landmark_models/opencv_LBF.yaml")
left_eye_idx  = list(range(36, 42))
right_eye_idx = list(range(42, 48))

image_folder = "anomal_hd_30fps_01"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg"))])
annotations = load_annotations("anomal_hd_30fps_01/anot.txt")

frames = []
for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    if frame is not None: frames.append(frame)

ear_window, correct, total = [], 0, 0
for idx, frame in enumerate(frames):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    final_prediction = "open"
    
    if len(faces) > 0:
        face_rect = faces[0]
        ok, landmarks_list = facemark.fit(gray, np.array([face_rect]))
        if ok:
            landmarks = np.reshape(landmarks_list[0], (-1, 2))
            for(x, y) in landmarks: cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            left_eye = landmarks[left_eye_idx]
            right_eye = landmarks[right_eye_idx]
            leftEAR = compute_ear(left_eye)
            rightEAR = compute_ear(right_eye)
            ear = (leftEAR + rightEAR) / 2.0
            
            ear_window.append(ear)
            if len(ear_window) > FRAME_WINDOW: ear_window.pop(0)
            avg_ear = np.mean(ear_window)
            
            (ex, ey, ew, eh) = cv2.boundingRect(left_eye.astype(np.int32))
            eye_roi = gray[ey:ey+eh, ex:ex+ew]
            lbp_mean = get_lbp_mean(eye_roi)
            
            if avg_ear < EAR_THRESHOLD and lbp_mean < LBP_THRESHOLD: final_prediction = "closed"
            else: final_prediction = "open"
    frame_number = idx + 1
    gt = "closed" if is_closed_ground_truth(frame_number, annotations) else "open"
    if final_prediction == gt: correct += 1
    total += 1

    cv2.putText(frame, f"Eye: {final_prediction} GT: {gt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 105, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
cv2.destroyAllWindows()

accuracy = (correct / total * 100) if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")