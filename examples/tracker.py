import numpy as np
import cv2
import pickle
from collections import deque

class Tracker():
    
    # When starting a new instance please be sure to specify all unassigned variables
    def __init__(self, Mycalib_width = 9, Mycalib_height = 6, Mywindow_width = 25, Mywindow_height = 80, Mymargin = 50, Myminpix = 50000, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15, Mybot_width = .76, Mymid_width = .08, Myheight_pct = .62, Mybottom_trim = .935):
        # the width of chessboard image used for camera calibration
        self.cal_width = Mycalib_width

        # the heigth of chessboard image used for camera calibration
        self.cal_height = Mycalib_height
        
        # the window pixel width of the center values, used to count pixels inside the center windows to determine curve values
        self.window_width = Mywindow_width

        # the window pixel height of the center values, used to count pixels inside center windows to determine curve values breaks the image into vertical levels
        self.window_height = Mywindow_height

        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = Mymargin

        # The pixel count needed at minimum to update window center 
        self.minpix = Myminpix

        # meters per pixel in vertical axis
        self.ym_per_pix = My_ym

        # meters per pixel in horizontal axis
        self.xm_per_pix = My_xm

        # to take average values based on previous frames
        self.smooth_factor = Mysmooth_factor

        # percent of bottom trapizoid height
        self.bot_width = Mybot_width

        # percent of middle trapizoid height
        self.mid_width = Mymid_width

        # percent for trapizoid height
        self.height_pct = Myheight_pct

        # percent from top to bottom to avoid car hood
        self.bottom_trim = Mybottom_trim

        # list that stores all the past (left, right) center set values used for smoothing the output
        self.recent_centers = deque(maxlen=100)             # memory from past 100 frames
        
        # total count of frames or images corresponding to data saved in memory
        self.total = len(self.recent_centers)

        # Read in the saved camera calibration data
        self.dist_pickle = None

        # Read in the saved calibration camera matrix
        self.mtx = None

        # Read in the saved calibration distortion coefficients
        self.dist = None

    def clear(self):
        """clear the tracker"""
        self.recent_centers.clear()
        self.total = 0

    def set_camera_calibration(self):
        """set camera calibration"""
        self.dist_pickle = pickle.load(open("../camera_cal/calibration_pickle.p", "rb"))
        self.mtx = self.dist_pickle["mtx"]
        self.dist = self.dist_pickle["dist"]

### DEBUG - Begins
#if False:
#    class Average:
#        """store list by adding to item and return Average"""
#        def __init__(self, period):
#            self.total = 0
#            self.queue = []
#            self.period = period
#        
#        def append(self, item):
#            """add item to the list"""
#            #print("append item: ", item)
#            self.total += item
#            self.queue.append(item)
#            #print("append total new ", self.total)
#            if len(self.queue) > self.period:
#                self.total -= self.queue.pop(0)
#            
#        def getAverage(self):
#            """get the mean of the list"""
#            length = len(self.queue)
#            if length == 0:
#                length = 1
#            #print("get avg, total: ", self.total, ", queue length: ", len(self.queue), "mean: ", self.total/length)
#            return self.total/length
#        
#        def clear(self):
#            """clear the average"""
#            self.total = 0
#            self.queue.clear()

#    class AveragePolyFit:
#        """calculate and store the average PolyFit for left and right poly fit"""
#        
#        def __init__(self, period):
#            self.left = Average(period)
#            self.right = Average(period)
#            
#        def clear(self):
#            """clear all averages"""
#            self.left.clear()
#            self.right.clear()
#        
#        def addLeft(self, item):
#            """add the left polyFit"""
#            self.left.append(item)
#        
#        def getAvgLeft(self):
#            """get the average left poly fit"""
#            return self.left.getAverage()
#        
#        def addRight(self, item):
#            """add the right polyFit"""
#            self.right.append(item)
#        
#        def getAvgRight(self):
#            """get the average right poly fit"""
#            return self.right.getAverage()
### DEBUG - Ends
