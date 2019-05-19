import cv2

import Preprocessing
import characterClass

def detectPlate(img):
    # Gray and threshold of image
    imgGrayscale, imgThresh = Preprocessing.preprocess(img)
    cv2.imshow('2. Threshold Image', imgThresh)
    cv2.imwrite('Threshold.jpg', imgThresh)
    
    # Contours containing possible characters
    possibleCharacters = findCharacters(imgThresh)
    
    # Create combination of number plates
    possiblePlates = findCharacterCombinations(possibleCharacters)
    
    # Return result
    return possiblePlates
    
    
def findCharacters(imgThresh):
    # Return value
    possibleCharacters = []
    
    # Total number of possible characters
    count = 0
    
    # Contours spresent in the imgThresh
    contours, hierarchy = cv2.findContours(imgThresh, 
                                           cv2.RETR_CCOMP, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    imgCopy = imgThresh.copy()
    cv2.drawContours(imgCopy, contours, -1, 255)
    
    for contour in contours:
        # Dimentions of contours
        [x, y, w, h] = cv2.boundingRect(contour)
        ar = w / h
        
        # Minimum required conditions for contours
        min_w = 8
        min_h = 16
        min_ar = 0.25
        max_ar = 0.85
        
        if w >= min_w and h >= min_h and ar >= min_ar and ar <= max_ar:
            cv2.rectangle(imgCopy, (x, y), (x+w, y+h), 255, 2)
            possibleCharacters.append(characterClass.characters(contour))
    
    cv2.imshow('3. Contours', imgCopy)
    cv2.imwrite('Contours.jpg', imgCopy)
    return possibleCharacters

def findCharacterCombinations(possibleCharacters):
    # Return value
    possiblePlates = []
    
    # Sort all contours based on X axis
    possibleCharacters.sort(key = lambda x: x.X)
    
    while len(possibleCharacters) > 0:
        plate = []
        plate.append(possibleCharacters.pop(0))
                
        for i in possibleCharacters:
            if (plate[-1].Y + (plate[-1].H/3) > i.Y and
                plate[-1].Y - (plate[-1].H/3) < i.Y and
                plate[-1].X + (3*plate[-1].H/2) > i.X and
                plate[-1].X + (plate[-1].W/2) < i.X and
                abs(i.H - plate[-1].H) < (i.H/10)):
                plate.append(i)
                
        possibleCharacters = [i for i in possibleCharacters if i not in plate]
                
        if len(plate) >= 3:
            possiblePlates.append(plate)
        
    return possiblePlates

        