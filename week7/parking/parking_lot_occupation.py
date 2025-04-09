import cv2, glob, os, time
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC

def lbp_histogram(img, P, R):
    lbp = local_binary_pattern(img, P, R, 'uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0,P+3), range=(0,P+2))
    hist = hist.astype(float) / (hist.sum() + 1e-6)
    return hist

def load_train_data(path, P, R):
    X, y = [], []
    for label, folder in enumerate(['free', 'full']):
        for img_path in glob.glob(f"{path}/{folder}/*.png"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hist = lbp_histogram(img, P, R)
            X.append(hist)
            y.append(label)
    return X, y

def train_svm(X, y):
    clf = LinearSVC(max_iter=10000)
    clf.fit(X, y)
    return clf

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def four_point_transform(image, coords):
    pts = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    rect = order_points(np.array(pts))
    width = int(max(np.linalg.norm(rect[2]-rect[3]), np.linalg.norm(rect[1]-rect[0])))
    height = int(max(np.linalg.norm(rect[1]-rect[2]), np.linalg.norm(rect[0]-rect[3])))
    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

def gauss_canny(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges.mean() > 5

def median_sobel(img):
    median = cv2.medianBlur(img, 5)
    sobel = cv2.Sobel(median, cv2.CV_64F, 1, 1)
    abs_sobel = cv2.convertScaleAbs(sobel)
    _, binary_sobel = cv2.threshold(abs_sobel, 50, 255, cv2.THRESH_BINARY)
    return binary_sobel.mean() > 5

def evaluate_method(method_func, spots, test_imgs, ground_truths):
    TP = TN = FP = FN = 0
    start = time.time()
    for img, gt in zip(test_imgs, ground_truths):
        for idx, coords in enumerate(spots):
            roi = four_point_transform(img, coords)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            detected = method_func(gray)
            actual = gt[idx]
            if detected == actual:
                if detected: TP += 1
                else: TN += 1
            else:
                if detected: FP += 1
                else: FN += 1
    duration = time.time() - start
    accuracy = (TP+TN)/(TP+TN+FP+FN)*100
    return accuracy, duration, TP, TN, FP, FN

def main():
    spots = [list(map(float, l.strip().split())) for l in open('parking_map_python.txt')]
    test_img_paths = sorted(glob.glob("test_images_zao/*.jpg"))
    test_imgs = [cv2.imread(p) for p in test_img_paths]
    ground_truths = [[int(l.strip()) for l in open(p.replace('.jpg','.txt'))] for p in test_img_paths]
    methods = {"Gauss+Canny": gauss_canny, "Median+Sobel": median_sobel}

    for P, R in [(8,1), (16,2), (24,3)]:
        X_train, y_train = load_train_data('free_full', P, R)
        clf = train_svm(X_train, y_train)
        methods[f"LBP P={P} R={R}"] = lambda img, clf=clf, P=P, R=R: clf.predict([lbp_histogram(img,P,R)])[0]

    for name, func in methods.items():
        accuracy, duration, TP, TN, FP, FN = evaluate_method(func, spots, test_imgs, ground_truths)
        print(f"{name} | Accuracy={accuracy:.2f}% | Time={duration:.2f}s | TP={TP}, TN={TN}, FP={FP}, FN={FN}")

if __name__ == '__main__': main()