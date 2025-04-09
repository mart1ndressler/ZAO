import cv2, glob, os, time
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

def lbp_histogram(img, P, R, fuse_edges=False):
    lbp = local_binary_pattern(img, P, R, 'uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, P+3), range=(0, P+2))
    hist = hist.astype(float) / (hist.sum() + 1e-6)
    
    if fuse_edges:
        blur = cv2.GaussianBlur(img, (5,5), 0)
        edges = cv2.Canny(blur, 80, 150)
        edges_mean = edges.mean()
        return np.hstack([hist, edges_mean])
    else: return hist

def load_train_data(path, P, R, fuse_edges=False):
    X, y = [], []
    for label, folder in enumerate(['free', 'full']):
        for img_path in glob.glob(f"{path}/{folder}/*.png"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            feat = lbp_histogram(img, P, R, fuse_edges=fuse_edges)
            X.append(feat)
            y.append(label)
    return X, y

def train_svm(X, y):
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=10000)
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
    edges = cv2.Canny(blur, 80, 150)
    return edges.mean() > 5

def median_sobel(img):
    median = cv2.medianBlur(img, 5)
    sobel = cv2.Sobel(median, cv2.CV_64F, 1, 1)
    abs_sobel = cv2.convertScaleAbs(sobel)
    _, binary_sobel = cv2.threshold(abs_sobel, 20, 255, cv2.THRESH_BINARY)
    return binary_sobel.mean() > 4

def evaluate_method(method_func, method_name, spots, test_img_paths, test_imgs, ground_truths):
    TP = TN = FP = FN = 0
    start = time.time()
    os.makedirs("results", exist_ok=True)

    for i, (img, gt) in enumerate(zip(test_imgs, ground_truths)):
        result_img = img.copy()
        base_name = os.path.basename(test_img_paths[i])

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

            pts = np.array(coords, np.int32).reshape((-1,1,2))
            color = (0,255,0) if detected == 0 else (0,0,255)
            cv2.polylines(result_img, [pts], True, color, 2)
            
            text_label = "free" if detected == 0 else "full"
            x_text = int(coords[0])
            y_text = int(coords[1]) - 10
            cv2.putText(result_img, text_label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        out_name = os.path.join("results", f"{method_name}_{base_name}")
        cv2.imwrite(out_name, result_img)   
        
    duration = time.time() - start
    accuracy = (TP+TN)/(TP+TN+FP+FN)*100
    return accuracy, duration, TP, TN, FP, FN

def main():
    spots = [list(map(float, l.strip().split())) for l in open('parking_map_python.txt')]
    test_img_paths = sorted(glob.glob("test_images_zao/*.jpg"))
    test_imgs = [cv2.imread(p) for p in test_img_paths]
    ground_truths = [[int(l.strip()) for l in open(p.replace('.jpg','.txt'))] for p in test_img_paths]
    methods = {"GaussCanny": gauss_canny, "MedianSobel": median_sobel}

    for P, R in [(8,1), (16,2), (24,3)]:
        X_train, y_train = load_train_data('free_full', P, R, fuse_edges=False)
        clf = train_svm(X_train, y_train)
        methods[f"LBP_P{P}R{R}"] = lambda img, c=clf, p=P, r=R: c.predict([lbp_histogram(img,p,r)])[0]

    for P, R in [(8,1), (16,2), (24,3)]:
        X_train_fuse, y_train_fuse = load_train_data('free_full', P, R, fuse_edges=True)
        clf_fuse = train_svm(X_train_fuse, y_train_fuse)
        methods[f"LBP_CannyFusion_P{P}R{R}"] = lambda img, c=clf_fuse, p=P, r=R: c.predict([lbp_histogram(img,p,r,fuse_edges=True)])[0]

    for method_name, method_func in methods.items():
        accuracy, duration, TP, TN, FP, FN = evaluate_method(method_func, method_name, spots, test_img_paths, test_imgs, ground_truths)
        print(f"{method_name}: Accuracy={accuracy:.2f}% | Time={duration:.2f}s | "f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        
if __name__ == '__main__': main()