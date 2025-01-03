import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms

def get_keypoints_and_descriptors(left_img, right_img):
    # Convert images to grayscale
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("grey_picture",gray_left)

    # Use SIFT for keypoints and descriptor extraction
    sift = cv2.SIFT_create()

    key_points1, descriptor1 = sift.detectAndCompute(gray_left, None)
    key_points2, descriptor2 = sift.detectAndCompute(gray_right, None)

    return key_points1, descriptor1, key_points2, descriptor2


def match_keypoints(descriptor1, descriptor2, key_points1, key_points2):
    # Use BFMatcher with the ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)  # euclidean distance
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good_matches = []  # m: The best match (descriptor with the smallest distance).
    for m, n in matches:  # n: The second-best match (descriptor with the second smallest distance).
        if m.distance < 0.75 * n.distance:  # The threshold 0.75 ensures the best match is at least 25% better than the second-best match.
            left_pt = key_points1[
                m.queryIdx].pt  # Retrieves the (x, y) coordinates of the keypoint in the first image corresponding to the match m.
            right_pt = key_points2[
                m.trainIdx].pt  # Retrieves the (x, y) coordinates of the keypoint in the second image corresponding to the match m.
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])

    return good_matches


def homography(points):
    Ar = []
    for pt in points:
        x, y = pt[0], pt[1]
        X, Y = pt[2], pt[3]
        Ar.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        Ar.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

    Ar = np.array(Ar)
    u, s, vh = np.linalg.svd(Ar)  # single variable ecomposition
    H = vh[-1, :].reshape(3, 3)
    return H / H[2, 2]


def ransac(matches, num_iterations=5000, threshold=5):
    if len(matches) < 4:
        print("Not enough matches to compute a homography.")
        return None  # Handle this case appropriately

    max_inliers = []
    best_H = None

    for _ in range(num_iterations):
        sampled_pts = random.sample(matches, 4)
        H = homography(sampled_pts)

        inliers = []
        for pt in matches:
            p1 = np.array([pt[0], pt[1], 1])  # converts coordinates(x,y)
            p2 = np.array([pt[2], pt[3], 1])  # to vertical matrix [x,y,1]
            projected_p1 = np.dot(H, p1)  # it applies homography to point p1
            projected_p1 /= projected_p1[
                2]  # To get the actual 2D coordinates in the second image, this vector is normalized by dividing by w

            error = np.linalg.norm(projected_p1[:2] - p2[:2])  # finding reprojection error
            if error < threshold:
                inliers.append(pt)

        if len(inliers) > len(max_inliers):  # larger inliners means good quality
            max_inliers = inliers
            best_H = H

    return best_H


def solution(left_img, right_img):
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoints_and_descriptors(left_img, right_img)
    good_matches = match_keypoints(descriptor1, descriptor2, key_points1, key_points2)
    # print(good_matches)
    final_H = ransac(good_matches)
    # final_H = np.array([
    #     [1.74684059e+00, -6.72057533e-02, -8.98561896e+02],
    #     [5.43868207e-01, 1.55613131e+00, -3.93404773e+02],
    #     [7.83932443e-04, 8.28567768e-06, 1.00000000e+00]
    # ])
    # print(final_H)

    # Get image dimensions
    rows1, cols1 = right_img.shape[:2]  #height and width of right image
    rows2, cols2 = left_img.shape[:2]  #height and width of left image
    # print(rows1)
    # print(cols1)

    # Calculate canvas size for panorama
    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    '''
    [[[  0,   0]],         # Top-left corner
     [[  0, rows1]],       # Bottom-left corner
     [[cols1, rows1]],     # Bottom-right corner
     [[cols1,   0]]]       # Top-right corner
    '''
    points2 = cv2.perspectiveTransform(points, final_H)  #is applying a geometric transformation (a perspective transformation) to the points using the homography matrix final_H.
    all_points = np.concatenate((points1, points2), axis=0) #The result is a unified array all_points containing all coordinate points from both input arrays.

    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # Translation matrix
    '''The result, H_translation, is the updated homography matrix that includes both
    the perspective transformation (final_H) and the translation. It ensures that all
    warped image points fit into the output canvas without negative coordinates.'''
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_H)

    # Warp left image to the panorama
    panorama = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
    panorama[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = right_img

    return panorama

#solution with backward interpolation
#
# def solution(left_img, right_img):
#     key_points1, descriptor1, key_points2, descriptor2 = get_keypoints_and_descriptors(left_img, right_img)
#     good_matches = match_keypoints(descriptor1, descriptor2, key_points1, key_points2)
#     # print(good_matches)
#     # final_H = ransac(good_matches)
#     final_H = np.array([
#         [1.74684059e+00, -6.72057533e-02, -8.98561896e+02],
#         [5.43868207e-01, 1.55613131e+00, -3.93404773e+02],
#         [7.83932443e-04, 8.28567768e-06, 1.00000000e+00]
#     ])
#     # print(final_H)
#
#     # Get image dimensions
#     rows1, cols1 = right_img.shape[:2]  # Height and width of right image
#     rows2, cols2 = left_img.shape[:2]  # Height and width of left image
#
#     # Calculate canvas size for panorama
#     points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
#     points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
#
#     points2 = cv2.perspectiveTransform(points, final_H)
#     all_points = np.concatenate((points1, points2), axis=0)
#
#     [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
#     [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
#
#     # Translation matrix
#     H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_H)
#
#     # Create an empty canvas for the panorama
#     canvas_height = y_max - y_min
#     canvas_width = x_max - x_min
#     panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
#
#     # Compute the inverse of the homography matrix
#     H_translation_inv = np.linalg.inv(H_translation)
#
#     # Perform backward warping with bilinear interpolation
#     for y in range(canvas_height):
#         for x in range(canvas_width):
#             # Map the canvas pixel to the source image using the inverse homography
#             src_coords = H_translation_inv.dot([x, y, 1])
#             src_coords /= src_coords[2]  # Normalize to get valid pixel coordinates
#
#             src_x, src_y = src_coords[0], src_coords[1]
#
#             # Check if the source coordinates are within the bounds of the source image
#             if 0 <= src_x < left_img.shape[1] and 0 <= src_y < left_img.shape[0]:
#                 # Perform bilinear interpolation
#                 x0, y0 = int(src_x), int(src_y)
#                 x1, y1 = min(x0 + 1, left_img.shape[1] - 1), min(y0 + 1, left_img.shape[0] - 1)
#
#                 a, b = src_x - x0, src_y - y0
#                 top = (1 - a) * left_img[y0, x0] + a * left_img[y0, x1]
#                 bottom = (1 - a) * left_img[y1, x0] + a * left_img[y1, x1]
#                 panorama[y, x] = (1 - b) * top + b * bottom
#
#     # Overlay the right image on the panorama
#     panorama[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = right_img
#
#     return panorama


# #solution with blending
# def blend_images(panorama, right_img, H_translation):
#     # Warp the right image to the panorama size
#     right_img_resized = cv2.warpPerspective(right_img, H_translation, (panorama.shape[1], panorama.shape[0]))
#
#     # Create a mask based on the horizontal blending range
#     height, width = panorama.shape[:2]
#     blend_mask = np.zeros((height, width), dtype=np.float32)
#
#     # Define the blending region (you could adjust this based on overlap region)
#     overlap_start = int(width * 0.25)  # Start of the overlap region
#     overlap_end = int(width * 0.75)  # End of the overlap region
#
#     # Create a linear blending mask (1 for left image, 0 for right image)
#     for x in range(overlap_start, overlap_end):
#         alpha = (x - overlap_start) / (overlap_end - overlap_start)
#         blend_mask[:, x] = alpha
#
#     # Blend the images
#     blended_img = np.zeros_like(panorama, dtype=np.uint8)
#     for y in range(height):
#         for x in range(width):
#             # Get the alpha for this pixel based on the mask
#             alpha = blend_mask[y, x]
#
#             # Blend pixel values for panorama (left image) and right image
#             left_pixel = panorama[y, x]
#             right_pixel = right_img_resized[y, x]
#
#             # Apply weighted average blending
#             blended_pixel = np.clip(left_pixel * (1 - alpha) + right_pixel * alpha, 0, 255).astype(np.uint8)
#
#             # Set blended pixel to the output image
#             blended_img[y, x] = blended_pixel
#
#     return blended_img


# def solution(left_img, right_img):
#     key_points1, descriptor1, key_points2, descriptor2 = get_keypoints_and_descriptors(left_img, right_img)
#     good_matches = match_keypoints(descriptor1, descriptor2, key_points1, key_points2)
#     # final_H = ransac(good_matches)
#     final_H = np.array([
#             [1.74684059e+00, -6.72057533e-02, -8.98561896e+02],
#             [5.43868207e-01, 1.55613131e+00, -3.93404773e+02],
#             [7.83932443e-04, 8.28567768e-06, 1.00000000e+00]
#         ])
#
#     # Get image dimensions
#     rows1, cols1 = right_img.shape[:2]
#     rows2, cols2 = left_img.shape[:2]
#
#     # Calculate canvas size for panorama
#     points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
#     points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
#     points2 = cv2.perspectiveTransform(points, final_H)
#     all_points = np.concatenate((points1, points2), axis=0)
#
#     [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
#     [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
#
#     # Translation matrix
#     H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(final_H)
#
#     # Warp left image to the panorama
#     panorama = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
#
#     # Blend the right image into the panorama using custom blending
#     panorama = blend_images(panorama, right_img, H_translation)
#
#     return panorama


# soultion with predefined cnn model



# Example of a simple image blending model using a pretrained network




# Function to ensure output directory exists
def create_output_folder(folder_name="output"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == "__main__":
    # Load images
    left_img = cv2.imread('pictures/pic1.1.jpg')
    right_img = cv2.imread('pictures/pic2.2.jpg')

    # Validate if images are loaded
    if left_img is None or right_img is None:
        print("Error loading images.")
        exit()

    # Generate panorama
    result_img = solution(left_img, right_img)

    # Create output folder and save the image
    output_folder = "output_images2"
    create_output_folder(output_folder)
    output_path = os.path.join(output_folder, "panorama_resultwithWarping.jpg")

    # Resize image to fit screen if needed
    result_img_resized = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)

    # Save the resized panorama image
    cv2.imwrite(output_path, result_img_resized)
    print(f"Panorama image saved at: {output_path}")

    # Display the resized panorama
    cv2.imshow('Resized Panorama', result_img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
