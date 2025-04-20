# import the necessary packages
import numpy as np
import cv2

class Stitcher:
    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images

        # Detect keypoints and extract features
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Match features between the two images
        M = self.matchKeypoints(kpsA, featuresA, kpsB, featuresB, ratio, reprojThresh)

        # If no matches are found, return None
        if M is None:
            return None

        # Apply homography transformation
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, 
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        return result

    def detectAndDescribe(self, image):
        """ Detect keypoints and extract SIFT features """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # FIX: Use SIFT correctly
        detector = cv2.SIFT_create()
        kps, features = detector.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, featuresA, kpsB, featuresB, ratio, reprojThresh):
        """ Match keypoints using KNN and find homography """
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        
        matches = []
        for m, n in rawMatches:
            if m.distance < ratio * n.distance:
                matches.append((m.trainIdx, m.queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None
