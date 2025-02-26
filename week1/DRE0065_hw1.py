import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, os
matplotlib.use("WebAgg")

paths = ["test_01.jpg","test_02.jpg","test_03.jpg","test_04.jpg"]
os.makedirs("cropped", exist_ok=True)
images, cropped_images = [], []

for i, f in enumerate(paths, 1):
    img = cv.imread(f)
    original = img.copy()
    
    for j in range(3):
        x1, y1, w, h = 778 + j*140, 725, 140, 220
        cv.line(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)
        cv.line(img, (x1 + w, y1), (x1, y1 + h), (0, 255, 0), 10)

        crop = original[y1:y1 + h, x1:x1 + w].copy()
        cv.imwrite(f"cropped/{i-1}{j+1}.png", crop)
        cropped_images.append(crop)
        print("SAVED TO:", f"cropped/{i-1}{j+1}.png")
    images.append(img)
resized = [cv.resize(im, (600, 300)) for im in images]

if len(resized) == 4:
    row1, row2 = cv.hconcat(resized[:2]), cv.hconcat(resized[2:])
    combined = cv.vconcat([row1, row2])
else: combined = cv.hconcat(resized)

cv.imshow("PARKING", combined)
rows, cols = 4, 3
cropped_mat = np.empty((rows, cols), dtype=object)

for idx, crop_bgr in enumerate(cropped_images):
    r, c = divmod(idx, cols)
    cropped_mat[r, c] = crop_bgr
    
fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
fig.suptitle("CROPPED PARKING SPACES")

for r in range(rows):
    for c in range(cols):
        block_bgr = cropped_mat[r, c]
        block_rgb = cv.cvtColor(block_bgr, cv.COLOR_BGR2RGB)
        axs[r, c].imshow(block_rgb)
        axs[r, c].axis("off")

plt.tight_layout()
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()