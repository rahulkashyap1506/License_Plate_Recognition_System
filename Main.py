import cv2
import numpy as np

import Plates

### Main Function ###
def main():    
    # Load training data
    Classifications = np.loadtxt("TrainingData/Classifications.txt", np.float32)
    FlattenedImages = np.loadtxt("TrainingData/FlattenedImages.txt", np.float32)
    
    Classifications = Classifications.reshape((Classifications.size, 1))
    
    # Define classsifier
    knn = cv2.ml.KNearest_create()
    
    # Train k-NN
    knn.setDefaultK(1)
    knn.train(FlattenedImages, cv2.ml.ROW_SAMPLE, Classifications)
    
    # Read image
    img = cv2.imread("5.jpg")
    if img == None:
        print("ERROR: Image is missing")
        return
    cv2.imshow('1. Original Image', img)
    
    # Search for possible plates in image
    possiblePlates = Plates.detectPlate(img)
    
    # Prepare image for testing
    img4rec = img.copy()
    img4rec = cv2.cvtColor(img4rec, cv2.COLOR_BGR2GRAY)
    img4rec = cv2.GaussianBlur(img4rec, (5, 5), 0)
    img4rec = cv2.adaptiveThreshold(img4rec, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    19,
                                    5)
    WIDTH, HEIGHT = 20, 30
    
    # List of string containing license plate numbers
    platesList = []
    # Testing loop
    for i in possiblePlates:
        string = ""
        for j in i:
            # Flattening the character
            imgChar = img4rec[j.Y:j.Y + j.H, j.X:j.X + j.W]            
            imgChar = cv2.resize(imgChar, (WIDTH, HEIGHT))
            imgChar = imgChar.reshape((1, WIDTH * HEIGHT))
            imgChar = np.float32(imgChar)
            
            # Testing
            ret, result, neighbors, dist = knn.findNearest(imgChar, k=1)
            
            Character = str(chr(int(result[0][0])))
            string = string + Character
        platesList.append(string)
            
    print(platesList)
    
    # Result preperation
    color = (255, 25, 25)
    for n, i in enumerate(possiblePlates):
        edge = 8
        # Create a rectangle around the plate
        cv2.rectangle(img, (i[0].X - edge, i[0].Y - edge),
                      (i[-1].X + i[-1].W + edge, i[-1].Y + i[-1].H + edge), 
                      color, 2)
        # Write license plate number on image
        cv2.putText(img, platesList[n], (i[0].X, i[0].Y - (2 * edge)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    
    # Show result
    cv2.imshow("4. Result", img)
    cv2.imwrite("Result.jpg", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


### Main Body ###
if __name__ == "__main__":
    main()
