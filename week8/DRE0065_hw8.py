import cv2
import os
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern

EAR_THRESHOLD, LBP_THRESHOLD, FRAME_WINDOW, CLOSED_FRAMES_THRESHOLD = 0.265, 5.0, 3, 5
consecutive_closed = 0

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

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
            if len(parts) == 2:
                intervals.append((int(parts[0]), int(parts[1])))
    return intervals

def is_closed_ground_truth(frame_idx, intervals):
    return any(start <= frame_idx <= end for start, end in intervals)

image_folder = "anomal_hd_30fps_01"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")])
annotations = load_annotations(os.path.join(image_folder, "anot.txt"))
frames = [cv2.imread(os.path.join(image_folder, f)) for f in image_files if cv2.imread(os.path.join(image_folder, f)) is not None]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

ear_window, correct, total = [], 0, 0
for idx, frame in enumerate(frames):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    final_prediction = "open"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        points = [(int(p.x * w), int(p.y * h)) for p in landmarks.landmark]

        left_eye = np.array([points[i] for i in LEFT_EYE_IDX])
        right_eye = np.array([points[i] for i in RIGHT_EYE_IDX])

        for(x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        leftEAR = compute_ear(left_eye)
        rightEAR = compute_ear(right_eye)

        if not (0.1 < leftEAR < 0.4 and 0.1 < rightEAR < 0.4): 
            continue

        ear_value = min(leftEAR, rightEAR)
        ear_window.append(ear_value)
        if len(ear_window) > FRAME_WINDOW:
            ear_window.pop(0)
        avg_ear = np.median(ear_window)

        (ex, ey, ew, eh) = cv2.boundingRect(left_eye.astype(np.int32))
        pad = 3
        ex, ey = max(0, ex - pad), max(0, ey - pad)
        ew += 2 * pad
        eh += 2 * pad
        eye_roi = cv2.cvtColor(frame[ey:ey+eh, ex:ex+ew], cv2.COLOR_BGR2GRAY)
        lbp_mean = get_lbp_mean(eye_roi)

        if avg_ear < EAR_THRESHOLD and lbp_mean < LBP_THRESHOLD:
            consecutive_closed += 1
        else:
            consecutive_closed = 0
        final_prediction = "closed" if consecutive_closed >= CLOSED_FRAMES_THRESHOLD else "open"

    frame_number = idx + 1
    gt = "closed" if is_closed_ground_truth(frame_number, annotations) else "open"
    if final_prediction == gt:
        correct += 1
    total += 1

    cv2.putText(frame, f"Eye: {final_prediction} GT: {gt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 105, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
accuracy = (correct / total * 100) if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")