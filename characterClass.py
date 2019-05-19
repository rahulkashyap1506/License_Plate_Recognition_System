import cv2

class characters:
    '''
    Class containing all possible characters of a number plate
    '''
    
    def __init__(self, contour):
        '''
        Constructor
        '''
        self.contour = contour        
        x, y, w, h = cv2.boundingRect(contour)
        self.X = x
        self.Y = y
        self.W = w
        self.H = h
