import cv2
import numpy as np

### Constants ###
MIN_AREA = 200
WIDTH = 20
HEIGHT = 30

### Main Function ###
def main():
    
    # Read character image
    imgTraining = cv2.imread("training_chars.png")
    
    # Check if image is present or not
    if imgTraining is None:
        print("ERROR: Training image is missing")
        return
    
    # Adding noise to characters
    imgGray = cv2.cvtColor(imgTraining, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
    
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      11, 
                                      2)
    
    # Separating individual characters
    contours, hierarchy = cv2.findContours(imgThresh, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Initial declarations for final results
    FlattenedImages = np.empty((0, HEIGHT * WIDTH))    
    Classifications = []
    
    # List of possible characters
    validChars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #validChars = "0123456789ABCDEFGHJKRSUWLMNP"
    intValidChars = []
    for i in validChars:
        intValidChars.append(ord(i))
    
    # Defining character classes
    for c in contours:
        if cv2.contourArea(c) > MIN_AREA:
            [X, Y, W, H] = cv2.boundingRect(c)
            
            cv2.rectangle(imgTraining, (X, Y), (X + W, Y + H), (255, 0, 0), 2)
            
            imgCrop = imgThresh[Y:Y + H, X:X + W]
            imgCropResized = cv2.resize(imgCrop, (WIDTH, HEIGHT))


            cv2.imshow("imageCrop", imgCrop)
            cv2.imshow("training_numbers.png", cv2.resize(imgTraining, (800, 500)))

            intChar = -1
            while intChar not in intValidChars:
                intChar = cv2.waitKey(0)
                if intChar >= 97:
                    intChar -= 32
        
            Classifications.append(intChar)                

            FlattenedImage = imgCropResized.reshape((1, WIDTH * HEIGHT))
            FlattenedImages = np.append(FlattenedImages, FlattenedImage, 0)
            
    Classifications = np.array(Classifications, np.float32)
    Classifications = Classifications.reshape(Classifications.size, 1)

    print("\n Training Data Created\n")
    
    # Saving final result
    np.savetxt("Classifications.txt", Classifications)
    np.savetxt("FlattenedImages.txt", FlattenedImages)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return
    
### Main Body ###
if __name__ == '__main__':
    main()