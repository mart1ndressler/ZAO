#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob
import os

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def main(argv):
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
   
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    
    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    print(pkm_coordinates)
    print("********************************************************")     
    
    occupancy_threshold = 0.05
    TP1 = TN1 = FP1 = FN1 = 0
    TP2 = TN2 = FP2 = FN2 = 0

    for img_path in test_images:
        img = cv2.imread(img_path)
        img_result1 = img.copy()
        img_result2 = img.copy()

        gt_path = img_path.replace('.jpg', '.txt')
        with open(gt_path, 'r') as gt_file:
            gt_lines = gt_file.readlines()

        for idx, spot in enumerate(pkm_coordinates):
            coords = list(map(float, spot[:8]))
            ground_truth = int(gt_lines[idx].strip())
            roi = four_point_transform(img, coords)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            blurred1 = cv2.GaussianBlur(gray, (5,5), 0)
            edges1 = cv2.Canny(blurred1, 50, 150)
            non_zero1 = cv2.countNonZero(edges1)
            ratio1 = non_zero1 / (roi.shape[0]*roi.shape[1])
            detected1 = 1 if ratio1 > occupancy_threshold else 0

            if ground_truth == 1 and detected1 == 1: TP1 += 1
            elif ground_truth == 0 and detected1 == 0: TN1 += 1
            elif ground_truth == 0 and detected1 == 1: FP1 += 1
            elif ground_truth == 1 and detected1 == 0: FN1 += 1
            
            pts = np.array(coords, np.int32).reshape((-1,1,2))
            color1 = (0,255,0) if detected1 == 0 else (0,0,255)
            cv2.polylines(img_result1, [pts], True, color1, 2)
            cv2.putText(img_result1, str(non_zero1), (int(coords[0]), int(coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            blurred2 = cv2.medianBlur(gray, 5)
            sobelx = cv2.Sobel(blurred2, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred2, cv2.CV_64F, 0, 1, ksize=3)
            sobel_abs = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))
            _, sobel_bin = cv2.threshold(sobel_abs, 50, 255, cv2.THRESH_BINARY)
            non_zero2 = cv2.countNonZero(sobel_bin)
            ratio2 = non_zero2 / (roi.shape[0]*roi.shape[1])
            detected2 = 1 if ratio2 > occupancy_threshold else 0
            
            if ground_truth == 1 and detected2 == 1: TP2 += 1
            elif ground_truth == 0 and detected2 == 0: TN2 += 1
            elif ground_truth == 0 and detected2 == 1: FP2 += 1
            elif ground_truth == 1 and detected2 == 0: FN2 += 1

            color2 = (0,255,0) if detected2 == 0 else (0,0,255)
            cv2.polylines(img_result2, [pts], True, color2, 2)
            cv2.putText(img_result2, str(non_zero2), (int(coords[0]), int(coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        os.makedirs("results", exist_ok=True)
        base_name = os.path.basename(img_path)

        result_name1 = os.path.join("results", f"result_gauss_canny_{base_name}")
        cv2.imwrite(result_name1, img_result1)
        print(f"Result (Gauss+Canny) image saved as: {result_name1}")
        
        result_name2 = os.path.join("results", f"result_median_sobel_{base_name}")
        cv2.imwrite(result_name2, img_result2)
        print(f"Result (Median+Sobel) image saved as: {result_name2}")

    total1 = TP1 + TN1 + FP1 + FN1
    accuracy1 = ((TP1 + TN1) / total1)*100 if total1 > 0 else 0
    print("***************** GAUSS + CANNY *****************")
    print(f"Accuracy: {accuracy1:.2f}%")
    print(f"TP: {TP1}, TN: {TN1}, FP: {FP1}, FN: {FN1}")
    print("*************************************************")

    total2 = TP2 + TN2 + FP2 + FN2
    accuracy2 = ((TP2 + TN2) / total2)*100 if total2 > 0 else 0
    print("**************** MEDIAN + SOBEL *****************")
    print(f"Accuracy: {accuracy2:.2f}%")
    print(f"TP: {TP2}, TN: {TN2}, FP: {FP2}, FN: {FN2}")
    print("************************************************")

if __name__ == "__main__": main(sys.argv[1:])