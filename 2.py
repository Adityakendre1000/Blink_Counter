# Importing necessary libraries
import cv2  # Importing OpenCV
import cvzone  # Importing cvzone library
from cvzone.FaceMeshModule import FaceMeshDetector  # Importing FaceMeshDetector from cvzone
from cvzone.PlotModule import LivePlot  # Importing LivePlot from cvzone

# Accessing the webcam
cap = cv2.VideoCapture(0)

# Initializing FaceMeshDetector with the maximum number of faces to be detected
detector = FaceMeshDetector(maxFaces=1)

# Creating a LivePlot for data visualization
plotY = LivePlot(640, 360, [18, 50], invert=True)

# Defining a list of facial landmarks for detecting blinks
idlist = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161]
ratiolist = []
blink_count = 0
prevratio = 0

# Infinite loop to continuously capture frames
while True:
    # Capturing a frame from the webcam
    success, img = cap.read()

    # Detecting the face mesh in the captured frame
    img, faces = detector.findFaceMesh(img, draw=False)

    # If a face is detected
    if faces:
        face = faces[0]
        for id in idlist:
            # Defining specific points for calculating the blink ratio
            leftup = face[159]
            leftdown = face[23]
            leftleft = face[130]
            leftright = face[243]

            # Calculating the lengths of vertical and horizontal lines
            lengthVer, _ = detector.findDistance(leftup, leftdown)
            lengthHor, _ = detector.findDistance(leftleft, leftright)

            # Calculating the ratio for blink detection
            ratio = ((lengthVer / lengthHor) * 100)

            # Tracking the blink count and updating the LivePlot
            if ratio < 30 and prevratio > 30:
                blink_count += 1

            # Displaying the blink count on the frame
            cvzone.putTextRect(img, f'Blink Count: {blink_count}', (100, 100))

            imgPlot = plotY.update(ratio)  # Updating the LivePlot with the ratio
            prevratio = ratio
            cv2.imshow("ImagePlot", imgPlot)  # Displaying the LivePlot
            cvzone.stackImages([img, imgPlot], 1, 1)  # Displaying images stacked horizontally

        else:
            cv2.imshow("ImagePlot", imgPlot)  # Displaying the LivePlot
            cvzone.stackImages([img, img], 1, 1)  # Displaying images stacked horizontally

    # Displaying the frame with facial landmarks
    cv2.imshow("Image", img)

    # Checking for the 'q' key press to quit the program
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
