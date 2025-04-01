import cv2
import time
import numpy as np

def is_eye_open_hough(eye_roi_bgr):
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=60, param2=20, minRadius=3, maxRadius=30)

    if circles is not None and len(circles) > 0: return True
    else: return False

def main():
    eye_states = []
    try:
        with open('eye-state.txt', 'r') as f:
            for line in f: eye_states.append(line.strip().lower())
    except Exception as e:
        print("Could not load 'eye-state.txt'!", e)
        eye_states = []

    face_cascade_frontal = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    face_cascade_profile = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier('eye_cascade_fusek.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
    cap = cv2.VideoCapture('fusek_face_car_01.avi')

    level_weight_threshold, frame_index, correct_predictions, total_predictions = 2.0, 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_time = time.time()

        try:
            faces_front, _, levelWeights_f = face_cascade_frontal.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100), maxSize=(500, 500), outputRejectLevels=True)
            faces_front = [bbox for bbox, lw in zip(faces_front, levelWeights_f) if lw > level_weight_threshold]
        except: faces_front = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100), maxSize=(500, 500))
        
        try:
            faces_prof, _, levelWeights_p = face_cascade_profile.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100), maxSize=(500, 500), outputRejectLevels=True)
            faces_prof = [bbox for bbox, lw in zip(faces_prof, levelWeights_p) if lw > level_weight_threshold]
        except: faces_prof = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100), maxSize=(500, 500))

        faces = list(faces_front) + list(faces_prof)
        detection_time = time.time() - start_time
        predicted_eye_state = "close"

        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15))
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                eye_roi_bgr = roi_color[ey:ey+eh, ex:ex+ew]
                
                if is_eye_open_hough(eye_roi_bgr):
                    predicted_eye_state = "open"
                    break
                
            mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(30, 30))
            for(mx, my, mw, mh) in mouths:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                break
            cv2.putText(frame, f"Eye: {predicted_eye_state}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if frame_index < len(eye_states):
            gt_state = eye_states[frame_index]
            if predicted_eye_state == gt_state: correct_predictions += 1
            total_predictions += 1

        frame_index += 1
        cv2.putText(frame, f"Detection Time: {detection_time:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Face/Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Overall eye state recognition accuracy: {accuracy * 100:.2f}%")
    else: print("eye-state.txt missing or empty!")

if __name__ == "__main__": main()