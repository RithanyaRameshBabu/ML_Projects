
"""## ID Tampering Detection

#### The purpose of this project is to detect tampering of ID cards using computer vision. This project will help organizations detect whether the ID provided by employees, customers, or anyone is original or has been tampered with.

#### For this project, we calculate the structural similarity between the original ID and the ID uploaded by the user to determine tampering.

"""

# import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import requests

!mkdir ID_card_tampering
!mkdir ID_card_tampering/image

# Open image and display
original = Image.open(r"/Emirates_ID_Card_For_UAE_Citizens_(New).png")
tampered = Image.open(r"/Emirates_ID_Card_For_UAE_Citizens_tampered.png")

"""#### Loading original and user provided images."""

# The file format of the source file.
print("Original image format : ",original.format)
print("Tampered image format : ",tampered.format)

# Image size, in pixels. The size is given as a 2-tuple (width, height).
print("Original image size : ",original.size)
print("Tampered image size : ",tampered.size)

"""#### Converting the format of  tampered image similar to original image."""

# Resize Image
original = original.resize((250, 160))
print(original.size)
original.save('ID_card_tampering/image/original.png')#Save image
tampered = tampered.resize((250,160))
print(tampered.size)
tampered.save('ID_card_tampering/image/tampered.png')#Saves image

"""####  Here, we checked the format and size of the original and tampered image."""

# Change image type if required from png to jpg
tampered = Image.open('ID_card_tampering/image/tampered.png')
tampered.save('ID_card_tampering/image/tampered.png')#can do png to jpg

"""#### Converting the size of tampered and original image."""

# Display original image
original

"""#### Orginial PAN card image used for comparision."""

# Display user given image
tampered

"""#### User provided image which will be compared with PAN card."""

# load the two input images
original = cv2.imread('ID_card_tampering/image/original.png')
tampered = cv2.imread('ID_card_tampering/image/tampered.png')

"""#### Reading images using opencv."""

# Convert the images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

"""#### Converting images into grayscale using opencv. Because in image processing many applications doesn't help us in identifying the important, edges of the coloured images also coloured images are bit complex to understand by machine beacuse they have 3 channel while grayscale has only 1 channel.  """

# Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

"""#### Structural similarity index helps us to determine exactly where in terms of x,y coordinates location, the image differences are. Here, we are trying to find similarities between the original and tampered image. The lower the SSIM score lower is the similarity."""

# Calculating threshold and contours
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

"""#### Here we are using the threshold function of computer vision which applies an adaptive threshold to the image which is stored in the form array. This function transforms the grayscale image into a binary image using a mathematical formula.
#### Find contours works on binary image and retrive the contours. This contours are a useful tool for shape analysis and recoginition. Grab contours grabs the appropriate value of the contours.
"""

# loop over the contours
for c in cnts:
    # applying contours on image
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)

"""#### Bounding rectangle helps in finding the ratio of width to height of bounding rectangle of the object. We compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images are different or not."""

#Diplay original image with contour
print('Original Format Image')
Image.fromarray(original)

#Diplay tampered image with contour
print('Tampered Image')
Image.fromarray(tampered)

#Diplay difference image with black
print('Different Image')
Image.fromarray(diff)

#Display threshold image with white
print('Threshold Image')
Image.fromarray(thresh)

"""### Summary

This project can be used by organizations where users are required to submit an ID for verification. It helps determine whether the provided ID is authentic or tampered. The system can be applied to any type of ID, such as Aadhaar, voter ID, passport, or driving license.

We calculated the structural similarity index (SSIM) to quantify differences between the original and user-provided images.
By applying thresholds and detecting contours on the grayscale binary images, we performed detailed shape analysis and recognition.
The SSIM score for the images is 0.7697, which is below the typical similarity threshold of 0.85. This indicates a high possibility of tampering.
Using SSIM as a guide, we can estimate the probability of the image being authentic:

SSIM ≥ 0.95 → Very likely authentic

0.85 ≤ SSIM < 0.95 → Possibly authentic

0.7 ≤ SSIM < 0.85 → Likely tampered

SSIM < 0.7 → Highly likely tampered

Finally, we visualized the differences and similarities by displaying the images along with contours, thresholded regions, and the computed difference map.
"""

