import cv2
import numpy as np 
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


# This program, given an image of a playing card, will return the rank and suit of that card.


# Loading the card and the images used to compare the card suit and rank
img = cv2.imread("testing_images\peightofh.JPG")
image = cv2.resize(img, (600, 800))
ace = cv2.imread("training_ranks\Ace.jpg")
ace = cv2.cvtColor(ace, cv2.COLOR_BGR2GRAY)
two = cv2.imread("training_ranks\Two.jpg")
two = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY)
three = cv2.imread("training_ranks\Three.jpg")
three = cv2.cvtColor(three, cv2.COLOR_BGR2GRAY)
four = cv2.imread("training_ranks\Four.jpg")
four = cv2.cvtColor(four, cv2.COLOR_BGR2GRAY)
five = cv2.imread("training_ranks\Five.jpg")
five = cv2.cvtColor(five, cv2.COLOR_BGR2GRAY)
six = cv2.imread("training_ranks\Six.jpg")
six = cv2.cvtColor(six, cv2.COLOR_BGR2GRAY)
seven = cv2.imread("training_ranks\Seven.jpg")
seven = cv2.cvtColor(seven, cv2.COLOR_BGR2GRAY)
eight = cv2.imread("training_ranks\Eight.jpg")
eight = cv2.cvtColor(eight, cv2.COLOR_BGR2GRAY)
nine = cv2.imread("training_ranks\kNine.jpg")
nine = cv2.cvtColor(nine, cv2.COLOR_BGR2GRAY)
ten = cv2.imread("training_ranks\Ten.jpg")
ten = cv2.cvtColor(ten, cv2.COLOR_BGR2GRAY)
jack = cv2.imread("training_ranks\Jack.jpg")
jack = cv2.cvtColor(jack, cv2.COLOR_BGR2GRAY)
queen = cv2.imread("training_ranks\Queen.jpg")
queen = cv2.cvtColor(queen, cv2.COLOR_BGR2GRAY)
king = cv2.imread("training_ranks\King.jpg")
king = cv2.cvtColor(king, cv2.COLOR_BGR2GRAY)
clubs = cv2.imread("training_suits\Clubs.jpg")
clubs = cv2.cvtColor(clubs, cv2.COLOR_BGR2GRAY)
spades = cv2.imread("training_suits\Spades.jpg")
spades = cv2.cvtColor(spades, cv2.COLOR_BGR2GRAY)
diamonds = cv2.imread("training_suits\Diamonds.jpg")
diamonds = cv2.cvtColor(diamonds, cv2.COLOR_BGR2GRAY)
hearts = cv2.imread("training_suits\Hearts.jpg")
hearts = cv2.cvtColor(hearts, cv2.COLOR_BGR2GRAY)


# Show the card we want to find the rank and suit of 
cv2.imshow('origional image', image)
cv2.waitKey(0)


# Create list of training images
list_of_suits = [clubs, spades, diamonds, hearts]
list_of_ranks = [ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, king] 


# Image Processing
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) 
blur = cv2.GaussianBlur(gray, (5, 5), 0) 
retval, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY) 

# Show the preprocessed image
cv2.imshow('processed image', thresh)
cv2.waitKey(0)


# Find and draw contours
edged = cv2.Canny(thresh, 30, 200)

# Show the contours found 
#cv2.imshow('contours', edged)
#cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(edged, contours, -1, (255, 0, 0), 5) 


# Find top left corner of card, enhance it, and split into rank and suit
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    x, y, w, h = cv2.boundingRect(c) 
pts = np.float32(approx)

# Manipulate the image so it is straight in order to get better image of rank and suit
# A lot of this perspective changer is taken from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def change_perspective(image, pts, w, h):
    temp_rect = np.zeros((4, 2), dtype = "float32")
    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    temp_rect[0] = tl
    temp_rect[1] = tr
    temp_rect[2] = br
    temp_rect[3] = bl
    maxWidth = 200
    maxHeight = 300
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1 ,maxHeight - 1],[0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped_image = cv2.cvtColor(warped_image,cv2.COLOR_BGR2GRAY)

    return warped_image


# Get the rank and suit from the straightened card by zooming in on top left corner 
warped_image = change_perspective(image, pts, w, h)
tl_corner = warped_image[0:84, 0:32]
corner_zoom = cv2.resize(tl_corner, (0, 0), fx = 4, fy = 4)
retval, card_threshold = cv2.threshold(corner_zoom, 160, 255, cv2.THRESH_BINARY_INV)
rank = card_threshold[20:185, 0:128]
suit = card_threshold[186:336, 0:128]
rank_contours, hier = cv2.findContours(rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rank_contours = sorted(rank_contours, key = cv2.contourArea, reverse = True)

if len(rank_contours) != 0:
    x1, y1, w1, h1 = cv2.boundingRect(rank_contours[0])
    rank_roi = rank[y1:y1 + h1, x1:x1 + w1]
    rank_sized = cv2.resize(rank_roi, (70, 125), 0, 0)
    rank_image = rank_sized

suit_contours, hier = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
suit_contours = sorted(suit_contours, key = cv2.contourArea, reverse = True)

if len(suit_contours) != 0:
    x2, y2, w2, h2 = cv2.boundingRect(suit_contours[0])
    suit_roi = suit[y2:y2 + h2, x2:x2 + w2]
    suit_sized = cv2.resize(suit_roi, (70, 100), 0, 0)
    suit_image = suit_sized

# Show the extracted rank and suit
cv2.imshow('warp', warped_image)
cv2.waitKey(0)
cv2.imshow('rank', rank_image)
cv2.waitKey(0)
cv2.imshow('suit', suit_image)
cv2.waitKey(0)


# Compare rank and suit to training images and select closest one
# This method of comparing images was found at https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mean_square_error(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]) 
    return err

list_of_ms = []
def compare_images(imageA, imageB, title):
    m = mean_square_error(imageA, imageB)
    list_of_ms.append(m)
    s = ssim(imageA, imageB)
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")

    # Show the comparison of each image to the training images (There's 17 so it takes a bit to go through all of them)
    # lt.show()


# Loop through ranks and suits and find closest one
for comp_rank in list_of_ranks:
    compare_images(rank_image, comp_rank, "Rank Comparison")

for comp_suit in list_of_suits:
    compare_images(suit_image, comp_suit, "Suit Comparison")

# Separate the list of m values into ranks and suits
list_of_suit_ms = [list_of_ms[13], list_of_ms[14], list_of_ms[15], list_of_ms[16]]
list_of_rank_ms = list_of_ms[0:13]

# Compare the extracted rank and suit to the training images, picks the smallest m value to find correct rank and suit and prints it onto origional image
if min(list_of_rank_ms) == list_of_rank_ms[0]:
    print("Ace")
    cv2.putText(image, 'Ace', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[1]:
    print("Two")
    cv2.putText(image, 'Two', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[2]:
    print("Three")
    cv2.putText(image, 'Three', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[3]:
    print("Four")
    cv2.putText(image, 'Four', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[4]:
    print("Five")
    cv2.putText(image, 'Five', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[5]:
    print("Six")
    cv2.putText(image, 'Six', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[6]:
    print("Seven")
    cv2.putText(image, 'Seven', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[7]:
    print("Eight")
    cv2.putText(image, 'Eight', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[8]:
    print("Nine")
    cv2.putText(image, 'Nine', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[9]:
    print("Ten")
    cv2.putText(image, 'Ten', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[10]:
    print("Jack")
    cv2.putText(image, 'Jack', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[11]:
    print("Queen")
    cv2.putText(image, 'Queen', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_rank_ms) == list_of_rank_ms[12]:
    print("King")
    cv2.putText(image, 'King', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

if min(list_of_suit_ms) == list_of_suit_ms[0]:
    print("of Clubs")
    cv2.putText(image, 'of Clubs', (285, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_suit_ms) == list_of_suit_ms[1]:
    print("of Spades")
    cv2.putText(image, 'of Spades', (285, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_suit_ms) == list_of_suit_ms[2]:
    print("of Diamonds")
    cv2.putText(image, 'of Diamonds', (285, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
elif min(list_of_suit_ms) == list_of_suit_ms[3]:
    print("of Hearts")
    cv2.putText(image, 'of Hearts', (285, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow("Final card", image)
cv2.waitKey(0)
