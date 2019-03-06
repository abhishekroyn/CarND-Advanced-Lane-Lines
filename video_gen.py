from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from tracker import tracker

import matplotlib.pyplot as plt         # for plotting durinig DEBUGGING only
import matplotlib.patches as patches    # for plotting durinig DEBUGGING only

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

frame_counter = 0                       # for plotting durinig DEBUGGING only

# Define a function that applies Sobel x or y
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_factor = np.max(gradmag)/255
    gradmag = (gradmag/scaled_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # Apply threshold 
    binary_output[(gradmag >= thresh[0]) & (gradmag >= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan2(sobely, sobelx))
        binary_output = np.zeros_like(absgraddir)
        # Apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output, s_binary, v_binary

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def get_roi(img, vertices):
    # Applies an image mask.
    # defining a blank mask to start with
    mask = np.zeros_like(img)   
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.array([vertices]), ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def process_image(img):
    
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # process image and generate binary pixel of interests
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 100))
    c_binary, s_binary, v_binary = color_threshold(img, sthresh=(100, 255),  vthresh=(150, 255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

    # Work on defining properties transformation area
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76 # percent of bottom trapizoid height
    mid_width = .08 # percent of middle trapizoid height
    height_pct = .62 # percent for trapizoid height
    bottom_trim = .935 # percent from top to bottom to avoid car hood
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    img_shape = img.shape
    roi_left_padding = img_size[0]*.20
    roi_right_padding = img_size[0]*.15
    roi_vertices =  np.array([
                        [roi_left_padding, 0],
                        [img_shape[1] - roi_right_padding, 0],
                        [img_shape[1] - roi_right_padding, img_shape[0]], 
                        [roi_left_padding, img_shape[0]]]
                        , dtype=np.int32)

    # Perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    # crop image
    warped_cropped = get_roi(warped, roi_vertices)

    window_width = 25
    window_height = 80
    margin = 50
    minpix = 50000

    # Set up the overall class to do all the tracking
    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = margin, Myminpix = minpix, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)

    window_centroids = curve_centers.find_window_centroids(warped_cropped)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and rigth windows
        l_points = np.zeros_like(warped_cropped)
        r_points = np.zeros_like(warped_cropped)

        # points used to find the left and right lanes
        rightx = []
        leftx = []

        # Go through each level and draw the Windows
        for level in range(0, len(window_centroids)):
            # add center value found in frame to the list of lane points per left, right
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])        
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped_cropped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped_cropped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points+l_points, np.uint8)    # add both left and rigth window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)    # make window pixels green
        warpage = np.array(cv2.merge((warped_cropped, warped_cropped, warped_cropped)), np.uint8)   # making the original road pixels 3 color channels
        result_raw = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)    # overlay the original road image with window results

    # If no window centers found, just display orginal road image
    else:
        result_raw = np.array(cv2.merge((warped_cropped,warped_cropped,warped_cropped)),np.uint8)  

    # fit the lane boundaries to the left, right center positions found
    yvals = range(0, warped_cropped.shape[0])

    res_yvals = np.arange(warped_cropped.shape[0]-(window_height/2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2, right_fitx[::-1]-window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

    ## DEBUG - Begins
    if False:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(42, 9))
        f.tight_layout()
        ax1.add_patch(patches.Polygon(src, fill=False))
        ax1.add_patch(patches.Polygon(dst, fill=False))
        ax1.imshow(result)
        ax1.set_title('result')
        ax2.add_patch(patches.Polygon(roi_vertices, fill=False, color='yellow'))
        ax2.imshow(warped, cmap='gray')
        ax2.set_title('warped')
        ax3.imshow(result_raw)
        ax3.set_title('result_raw')

        # Go through each level and draw the Windows
        for level in range(0, len(window_centroids)):
            x_l = window_centroids[level][0]
            x_r = window_centroids[level][1]
            y = img.shape[0]-((level + 0.5) * window_height)
            plt.plot(x_l, y, 'ro')
            plt.plot(x_r, y, 'bo')

        if False:
            f, (ax1a, ax1b, ax2a, ax2b, ax2, ax3a, ax3b, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 12, figsize=(42, 9))
            f.tight_layout()

            ax1a.add_patch(patches.Polygon(src, fill=False))
            ax1a.add_patch(patches.Polygon(dst, fill=False))
            ax1a.imshow(img)
            ax1a.set_title('img')
            ax1b.imshow(cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR))
            ax1b.set_title('original_image_warped_rgb')

            ax2a.imshow(s_binary, cmap='gray')
            ax2a.set_title('s_binary')
            ax2b.imshow(v_binary, cmap='gray')
            ax2b.set_title('v_binary')
            ax2.imshow(c_binary, cmap='gray')
            ax2.set_title('c_binary')

            ax3a.imshow(gradx, cmap='gray')
            ax3a.set_title('gradx')
            ax3b.imshow(grady, cmap='gray')
            ax3b.set_title('grady')

            ax4.add_patch(patches.Polygon(src, fill=False))
            ax4.add_patch(patches.Polygon(dst, fill=False))
            ax4.imshow(preprocessImage, cmap='gray')
            ax4.set_title('preprocessImage')

            ax5.add_patch(patches.Polygon(roi_vertices, fill=False, color='yellow'))
            ax5.imshow(warped, cmap='gray')
            ax5.set_title('warped')

            ax6.imshow(warped_cropped, cmap='gray')
            ax6.set_title('warped_cropped')

            ax7.add_patch(patches.Polygon(src, fill=False))
            ax7.add_patch(patches.Polygon(dst, fill=False))
            ax7.imshow(result)
            ax7.set_title('result')

            ax8.imshow(result_raw, cmap='gray')
            ax8.set_title('result_raw')

            # Go through each level and draw the Windows
            for level in range(0, len(window_centroids)):
                x_l = window_centroids[level][0]
                x_r = window_centroids[level][1]
                y = img.shape[0]-((level + 0.5) * window_height)
                plt.plot(x_l, y, 'ro')
                plt.plot(x_r, y, 'bo')

        # plt.show()
        
        global frame_counter 
        frame_counter = frame_counter + 1
        write_name = './debug_dir/video_frame_'+str(frame_counter)+'.png'
        plt.savefig(write_name)
    ## DEBUG - Ends

    ## DEBUG - Begins
    if False:
        global frame_counter 
        frame_counter = frame_counter + 1
        write_name = './debug_dir/video_frame_'+str(frame_counter)+'.png'
        plt.savefig(write_name)
    ## DEBUG - Ends

    ym_per_pix = curve_centers.ym_per_pix   # meters per pixel in y dimension
    xm_per_pix = curve_centers.xm_per_pix   # meters per pixel in x dimension

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text showing curvature, offset, and speed
    cv2.putText(result, 'Radius of Curvature = '+ str(round(curverad, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result

Output_video = 'output1_tracked.mp4'
Input_video = 'project_video.mp4'
#Output_video = 'output2_tracked.mp4'
#Input_video = 'challenge_video.mp4'
#Output_video = 'output3_tracked.mp4'
#Input_video = 'harder_challenge_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
video_clip.write_videofile(Output_video, audio=False)
