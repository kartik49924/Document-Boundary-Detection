########################################
# FOR USAGE: RUN BELOW COMMAND         #
# python scan.py -i images/m1.jpg #
########################################

import os
import numpy as np
import argparse
import cv2
import imutils


def order_points(pts):
	# initialize a list of coordinates that will be ordered
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

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
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

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx*scale, cy*scale]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled



def get_edges(image_list):
	### construct the argument parser and parse the arguments

	for imag in image_list:
	#loading image
		image = cv2.imread(imag)

		# Compute the ratio of the old height to the new height, clone it, 
		# and resize it easier for compute and viewing
		ratio = image.shape[0] / 500.0
		orig = image.copy()
		#print(image.shape)
		image = imutils.resize(image, height = 500)
		#print(image.shape)
		#cv2.imshow('My name is kartik',image)
		#image = cv2.resize(image, (image.shape[0]//10,image.shape[1]//10))

		### convert the image to grayscale, blur it, and find edges in the image

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Gaussian Blurring to remove high frequency noise helping in
		# Contour Detection 
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		# Canny Edge Detection
		edged = cv2.Canny(gray, 10, 300)


		print("STEP 1: Edge Detection")
		# cv2.imshow("Image", image)
		#cv2.imshow("Edged", edged)

		# finding the contours in the edged image, keeping only the
		# largest ones, and initialize the screen contour
		cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		## What are Contours ?
		## Contours can be explained simply as a curve joining all the continuous
		## points (along the boundary), having same color or intensity. 
		## The contours are a useful tool for shape analysis and object detection 
		## and recognition.

		# Handling due to different version of OpenCV
		#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts=cnts[0]
		# Taking only the top 5 contours by Area
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

		### Heuristic & Assumption

		# A document scanner simply scans in a piece of paper.
		# A piece of paper is assumed to be a rectangle.
		# And a rectangle has four edges.
		# Therefore use a heuristic like : we’ll assume that the largest
		# contour in the image with exactly four points is our piece of paper to 
		# be scanned.
		#screenCnt=[]
		# looping over the contours
		for c in cnts:
			### Approximating the contour

			#Calculates a contour perimeter or a curve length
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.01 * peri, True)#0.02

			# if our approximated contour has four points, then we
			# can assume that we have found our screen
			screenCnt = approx
			if len(approx) == 4:
				screenCnt = approx
				break
			
		# show the contour (outline) 
		print("STEP 2: Finding Boundary")
		#cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
		orig1=orig.copy()
		cv2.drawContours(orig1, [scale_contour(screenCnt,ratio)], -1, (0,0,255), 10)
		#cv2.imshow('mum',orig1)
		#cv2.imshow()
		#orig=cv2.resize(orig,(image.shape[1],image.shape[0]))
		#print(orig.shape,image.shape)
		#cv2.imshow("Boundary", cv2.resize(np.hstack((orig1,orig)),(orig1.shape[1]//3,orig1.shape[0]//6))) # to help in visualization
		#print(imag)
		#image=np.hstack((image,orig))
		cv2.imwrite('./output_images/{}_output.jpg'.format(imag[-7:-4]),orig1)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
imag=[]
#print(args)
if args['image'] is not None:
	imag.append(args['image'])
else:
	imag=os.listdir('./input_images/')
	for i in range(len(imag)):
		imag[i]=os.path.join('./input_images/',imag[i])
#print(imag)
get_edges(imag)


