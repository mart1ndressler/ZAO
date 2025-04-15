import cv2, time, numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC

def lbp_histogram(img, P, R):
    lbp = local_binary_pattern(img, P, R, 'uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2))
    return hist.astype(float) / (hist.sum() + 1e-6)

def prepare_training_data(video_file, cascade, P, R, states_file):
    cap = cv2.VideoCapture(video_file)
    eye_states = [line.strip().lower() for line in open(states_file)]
    X, y, frame_idx = [], [], 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(eye_states): break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cascade.detectMultiScale(gray, 1.05, 3, minSize=(15, 15))

        for(x, y_eye, w, h) in eyes:
            roi = gray[y_eye:y_eye+h, x:x+w]
            hist = lbp_histogram(roi, P, R)
            X.append(hist)
            label = 1 if eye_states[frame_idx] == "open" else 0
            y.append(label)
            break
        frame_idx += 1
    cap.release()
    return X, y

def train_classifier(X, y):
    clf = LinearSVC(max_iter=10000)
    clf.fit(X, y)
    return clf

def is_eye_open_hough(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=60, param2=20, minRadius=3, maxRadius=30)
    return circles is not None and len(circles) > 0

def evaluate_eye_state(video_file, cascade, clf, P, R, eye_states):
    cap = cv2.VideoCapture(video_file)
    correct, total = 0, 0
    start_time = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(eye_states): break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cascade.detectMultiScale(gray, 1.05, 3, minSize=(15, 15))
        prediction = "close"

        for(x, y_eye, w, h) in eyes:
            roi_gray = gray[y_eye:y_eye+h, x:x+w]
            roi_bgr = frame[y_eye:y_eye+h, x:x+w]

            hist = lbp_histogram(roi_gray, P, R)
            lbp_pred = clf.predict([hist])[0]

            hough_pred = 1 if is_eye_open_hough(roi_bgr) else 0
            final_pred = "open" if (lbp_pred + hough_pred) >= 1 else "close"
            prediction = final_pred
            break

        if prediction == eye_states[frame_idx]: correct += 1
        total += 1
        frame_idx += 1
    duration = time.time() - start_time
    accuracy = correct / total * 100 if total else 0
    cap.release()
    return accuracy, duration

def main():
    video_file = 'fusek_face_car_01.avi'
    cascade = cv2.CascadeClassifier('eye_cascade_fusek.xml')
    eye_states = [line.strip().lower() for line in open('eye-state.txt')]

    configurations = [(8,1), (16,2), (24,3)]
    for P, R in configurations:
        print(f"Configuration LBP+HoughCircles: P={P}, R={R}")

        X_train, y_train = prepare_training_data(video_file, cascade, P, R, 'eye-state.txt')
        clf = train_classifier(X_train, y_train)
        accuracy, duration = evaluate_eye_state(video_file, cascade, clf, P, R, eye_states)
        print(f"Accuracy = {accuracy:.2f}% | Time = {duration:.2f}s\n")

if __name__ == '__main__': main()