import streamlit as st
import numpy as np
import cv2

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7)    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)    
    return opening

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def perspective_correction(image, threshold_value=100, min_line_length=100, max_line_gap=10):
    processed_image = preprocess_image(image)
    lines = cv2.HoughLinesP(
        processed_image,
        rho=1,
        theta=np.pi/180,
        threshold=threshold_value,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if len(lines) < 1:
        st.write("No lines found.")
        return image

    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

    if len(endpoints) >= 4:
        endpoints = np.array(endpoints)
        corrected_image = four_point_transform(image, endpoints)
    else:
        st.write("Insufficient endpoints found.")
        return image

    return corrected_image

def perspective_correction2(image, threshold_value=100, min_line_length=100, max_line_gap=10):
    processed_image = preprocess_image(image)
    lines = cv2.HoughLinesP(
        processed_image,
        rho=1,
        theta=np.pi/180,
        threshold=threshold_value,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if len(lines) < 1:
        st.write("No lines found.")
        return image

    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

    if len(endpoints) >= 4:
        endpoints = np.array(endpoints)
        hull = cv2.convexHull(endpoints)
        rect = np.zeros((4, 2), dtype=np.float32)
        hull = hull.reshape(-1, 2)
        rect[0] = hull[np.argmin(hull.sum(axis=1))]
        rect[2] = hull[np.argmax(hull.sum(axis=1))]
        rect[1] = hull[np.argmin(np.diff(hull, axis=1))]
        rect[3] = hull[np.argmax(np.diff(hull, axis=1))]
        widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
        M, _ = cv2.findHomography(rect, dst)
        corrected_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    else:
        st.write("Insufficient endpoints found.")
        return image

    return corrected_image