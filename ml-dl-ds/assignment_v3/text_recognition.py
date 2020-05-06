# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image data-set/ms-7.jpg --padding 0.1

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
from cv2 import cv2
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=1120,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=1120,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.1,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
(thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
image = cv2.bilateralFilter(image,9,75,75)

# Remove horizontal and vertical lines
horizontal_img=255-image
vertical_img=255-image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

mask_img=vertical_img+horizontal_img
image=np.bitwise_or(image,mask_img)
(origH, origW) = orig.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image2 = cv2.resize(image, (newW, newH))
(H, W) = image2.shape[:2]
cv2.imwrite("detecttable.jpg",image2)
# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

image3 = np.zeros((H,W,3))
for ch in range(3):
	for xx in range(W):
		for yy in range(H):
			image3[xx,yy,ch]=image2[xx,yy]
image2 = 255 - image3

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
# 	(123.68, 116.78, 103.94), swapRB=True, crop=False)
blob = cv2.dnn.blobFromImage(np.float32(image2), 1.0, (W, H), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)
# print (boxes)
# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# in order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	# apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# extract the actual padded ROI
	roi = image[startY:endY, startX:endX]

	# in order to apply Tesseract v4 to OCR text we must supply
	# (1) a language, (2) an OEM flag of 4, indicating that the we
	# wish to use the LSTM neural net model for OCR, and finally
	# (3) an OEM value, in this case, 7 which implies that we are
	# treating the ROI as a single line of text
	config = ("-l eng --oem 1 --psm 8")
	text = pytesseract.image_to_string(roi, config=config)

	# add the bounding box coordinates and OCR'd text to the list
	# of results
	results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][0])
results = sorted(results, key=lambda r:r[0][1] )
print (results[:5])
# loop over the results
Score={}
for ((startX, startY, endX, endY), text) in results:
	#display the text OCR'd by Tesseract
	# print("OCR TEXT")
	# print("========")
	idx=results.index(((startX, startY, endX, endY), text))
	text= text.strip()
	text = text.replace("|","")
	text = text.replace("\"","")
	text = text.replace("'","")
	text = text.replace(" ","")
	results[idx]=((startX, startY, endX, endY), text)
	# print (text)
	# print("{}".format(text))
for ((startX, startY, endX, endY), text) in results:
	if ("MATHEMATICS" in text) or ("PHYSICS" in text) or ("CHEMISTRY" in text):
		print(((startX, startY, endX, endY), text))
		# in order to obtain a better OCR of the text we can potentially
		# apply a bit of padding surrounding the bounding box -- here we
		# are computing the deltas in both the x and y directions
		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * args["padding"])

		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		ROI = unsharp_mask(orig[startY:endY, startX:(origW-startX)])
		# ROI = cv2.GaussianBlur(ROI,(5,5),0)
		ROI_H, ROI_W = ROI.shape[:2]
		ROI = cv2.resize(ROI,(int(0.9*ROI_W),int(1.2*ROI_H)),interpolation=cv2.INTER_CUBIC)
		cv2.rectangle(image, (startX, startY), (origW-startX, endY),(0, 0, 255), 1)
		text_result = pytesseract.image_to_string(ROI, config="-l eng --oem 1 --psm 7")
		text_result = ''.join(e for e in text_result if e.isalnum() or e==" ").strip()
		print(text_result)
		line = text_result.split(" ")
		for i in range(len(line)):
			if line[-1-i].isnumeric():
				Score[line[0]]=line[-1-i]
				break

final_score = (int(Score["MATHEMATICS"]) + int(Score["PHYSICS"]) + int(Score["CHEMISTRY"]))/3
if final_score >= 90:
	print ("Campus-Alpha")
elif final_score>=80 :
	print ("Campus-Beta")
elif final_score>=70 :
	print ("Campus-Gama")
else:
	print ("Better luck next time")
	
cv2.imwrite("detectable_text.jpg",image)