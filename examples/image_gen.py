#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import numpy as np
import cv2
import glob
from tracker import Tracker

## DEBUG - Begins
if False:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from tracker import Average
    from tracker import AveragePolyFit
## DEBUG - Ends

## DEBUG - Begins
if False:
    frame_counter = 0
## DEBUG - Ends

## DEBUG - Begins
if False:
    avgPolyFit = AveragePolyFit(1)            # 1 for single image and 15 for video frames
    avgPolyFit.clear()
## DEBUG - Ends

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

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
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

def preprocess_image(img, mtx, dist):
    # Read in image
    img = cv2.imread(img)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # Calculate image size
    img_size = (img.shape[1], img.shape[0])

    # process image and generate binary pixel of interests
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 100))
    c_binary, s_binary, v_binary = color_threshold(img, sthresh=(100, 255), vthresh=(150, 255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255
    return preprocessImage, img, img_size

def find_src_dst_roi_vertices(bot_width, mid_width, height_pct, bottom_trim, img_size):
    # Define properties transformation area
    src = np.float32([[img_size[0]*(.5-mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+bot_width/2), img_size[1]*bottom_trim],[img_size[0]*(.5-bot_width/2), img_size[1]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    roi_left_padding = img_size[0]*.20
    roi_right_padding = img_size[0]*.15
    roi_vertices =  np.array([
                        [roi_left_padding, 0],
                        [img_size[0] - roi_right_padding, 0],
                        [img_size[0] - roi_right_padding, img_size[1]], 
                        [roi_left_padding, img_size[1]]]
                        , dtype=np.int32)
    return src, dst, roi_vertices

def find_perspective_transform(src, dst, preprocessImage, img_size):
    # Perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv, M

def get_roi(img, vertices):
    # Applies an image mask
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.array([vertices]), ignore_mask_color)
    
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_window_centroids(warped, window_width, window_height, margin, minpix, recent_centers, smooth_factor):
    window = np.ones(window_width)   # Create our window template that we will use for convolutions
    window_centroids = []            # Store the (left, right) window centroid positions per level

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(warped.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # for DEBUGGING only
    if False:
        leftx, rightx = [], []
        leftx.append(l_center)
        rightx.append(r_center)
    ## DEBUG - Ends

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of the window, not center of window
        offset = window_width/2        
        
        # Find the left centroid by using past right center as a reference
        l_min_index = int(max(l_center+offset-margin, 0))
        l_max_index = int(min(l_center+offset+margin, warped.shape[1]))

        l_signal = np.max(conv_signal[l_min_index:l_max_index])
        if l_signal > minpix:    # If total pixels > minpix pixels, recenter next window on their highest convolution-value position else center value remains same as the last value
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin, 0))
        r_max_index = int(min(r_center+offset+margin, warped.shape[1]))

        r_signal = np.max(conv_signal[r_min_index:r_max_index])
        if r_signal > minpix: # If total pixels > minpix pixels, recenter next window on their highest convolution-value position
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

        ## DEBUG - Begins
        if False:
            if l_signal < minpix:
                print ('left')
                print ('level - ', str(level))
                print (np.argmax(conv_signal[l_min_index:l_max_index]))
                print (np.max(conv_signal[l_min_index:l_max_index]))

            if r_signal < minpix:
                print ('right')
                print ('level - ', str(level))
                print (np.argmax(conv_signal[r_min_index:r_max_index]))
                print (np.max(conv_signal[r_min_index:r_max_index]))
        ## DEBUG - Ends

        ## DEBUG - Begins
        if False:
            l_signal_arg = np.argmax(conv_signal[l_min_index:l_max_index])
            if l_signal_arg < 2:    # not able to detect any hot pixel, so relying on mean
                leftx_mean = np.mean(leftx)        # getting the mean of all left pixels so far
                l_center = leftx_mean
            else:
                l_center = l_signal_arg + l_min_index - offset

            if l_signal_arg < 2:
                print ('left')
                print ('level - ', str(level))
                print (np.argmax(conv_signal[l_min_index:l_max_index]))
                print (np.max(conv_signal[l_min_index:l_max_index]))

            r_signal_arg = np.argmax(conv_signal[r_min_index:r_max_index])
            if r_signal_arg < 2:    #not able to detect any hot pixel, so relying on mean
                rightx_mean = np.mean(rightx)      # getting the mean of all right pixels so far
                r_center = rightx_mean
            else:
                r_center = r_signal_arg + r_min_index - offset

            if r_signal_arg < 2:
                print ('right')
                print ('level - ', str(level))
                print (np.argmax(conv_signal[r_min_index:r_max_index]))
                print (np.max(conv_signal[r_min_index:r_max_index]))
        ## DEBUG - Ends

        ## DEBUG - Begins
        if False:
            # There are many frames where the convolve identifies wrong hot pixels. Checking mean and correcting any big deviation
            if np.absolute(l_center - leftx_mean) > window_width * 2:
                l_center = leftx_mean + (l_center - leftx_mean) * .7
                print ('left - level -', str(level))
            if np.absolute(r_center - rightx_mean) > window_width * 2:
                r_center = rightx_mean + (r_center - rightx_mean) * .7
                print ('right - level -', str(level))
        ## DEBUG - Ends

        # for DEBUGGING only
        if False:
            leftx.append(l_center)
            rightx.append(r_center)
        ## DEBUG - Ends

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    recent_centers.append(window_centroids)

    ## DEBUG - Begins
    if False:
        # test results manually by printing & comparing
        avg_val = np.average(np.array(recent_centers)[-smooth_factor:], axis=0)
        print ("DEBUGGING \n")
        print (np.array(recent_centers))
        print ('\n')
        print (avg_val)
        print ('\n')
    ## DEBUG - Ends

    # return averaged values of the line centers, helps to keep the markers from jumping around too much
    return np.average(np.array(recent_centers)[-smooth_factor:], axis=0), recent_centers

def find_lane_mask_pixels(window_width, window_height, warped_cropped, window_centroids):
    # points used to find the left and right lanes based on window centroids
    leftx = []
    rightx = []

    # Points used to draw all the left and rigth windows
    l_points = np.zeros_like(warped_cropped)
    r_points = np.zeros_like(warped_cropped)

    # If we found any window centers
    if len(window_centroids) > 0:
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
    return leftx, rightx, l_points, r_points

def draw_results_left_right_window(l_points, r_points, warped_cropped, window_centroids):
    # If we found any window centers
    if len(window_centroids) > 0:
        template = np.array(l_points + r_points, np.uint8)    # add both left and rigth window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)    # make window pixels green
        warpage = np.array(cv2.merge((warped_cropped, warped_cropped, warped_cropped)), np.uint8)   # making the original road pixels 3 color channels
        result_raw = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)    # overlay the original road image with window results

    # If no window centers found, just display orginal road image
    else:
        result_raw = np.array(cv2.merge((warped_cropped,warped_cropped,warped_cropped)),np.uint8)
    return result_raw

def find_lane_boundaries(warped_cropped, window_height, leftx, rightx, window_width, l_points, r_points):

    yvals = range(0, warped_cropped.shape[0])
    res_yvals = np.arange(warped_cropped.shape[0]-(tracker.window_height/2), 0, -tracker.window_height)

    ## DEBUG - Begins
    if False:
        # points used to find the left and right lanes based on window mask regions        
        lefts = []
        rigths = []

        # points used to find the left and right lanes based on window mask regions        
        lefts = l_points.nonzero()
        rigths = r_points.nonzero()
        # Add current polyfit to average and then get the average to be used here
        avgPolyFit.addLeft(np.polyfit(lefts[0], lefts[1], 2))
        left_fit = avgPolyFit.getAvgLeft()

        #Add current polyfit to average and then get the average to be used here
        avgPolyFit.addRight(np.polyfit(rights[0], rights[1], 2))
        right_fit = avgPolyFit.getAvgRight()
    # for DEBUGGING only

    left_fit = np.polyfit(res_yvals, leftx, 2)
    right_fit = np.polyfit(res_yvals, rightx, 2)

    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-tracker.window_width/2, left_fitx[::-1]+tracker.window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-tracker.window_width/2, right_fitx[::-1]+tracker.window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+tracker.window_width/2, right_fitx[::-1]-tracker.window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    return left_lane, right_lane, inner_lane, yvals, res_yvals, left_fitx, right_fitx

def draw_results_left_right_inner_lines(img, left_lane, right_lane, inner_lane, Minv, img_size):
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
    return result

def calculate_curvature_offset(res_yvals, yvals, leftx, rightx, ym_per_pix, xm_per_pix, left_fitx, right_fitx, warped):
    # calculate radius of left curvature
    curve_fit_cr_left = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
    curverad_left = ((1 + (2*curve_fit_cr_left[0]*yvals[-1]*ym_per_pix + curve_fit_cr_left[1])**2)**1.5) / np.absolute(2*curve_fit_cr_left[0])

    # calculate radius of right curvature
    curve_fit_cr_right = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)*xm_per_pix, 2)
    curverad_right = ((1 + (2*curve_fit_cr_right[0]*yvals[-1]*ym_per_pix + curve_fit_cr_right[1])**2)**1.5) / np.absolute(2*curve_fit_cr_right[0])

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    return curverad_left, curverad_right, center_diff, side_pos

def draw_curvature_offset(result, curverad_left, curverad_right, center_diff, side_pos):
    cv2.putText(result, 'Radius of left Curvature = '+ str(round(curverad_left, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Radius of right Curvature = '+ str(round(curverad_right, 3)) + '(m)', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return result

## DEBUG - Begins
def draw_details_debugging_results(src, dst, result, result_raw, warped, warped_cropped, window_centroids, img, M, img_size, s_binary, v_binary, c_binary, gradx, grady, window_height):
    if False: 
        # draw the details results
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
        
        if False:
            global frame_counter 
            frame_counter = frame_counter + 1
            write_name = '../debug_dir/video_frame_'+str(frame_counter)+'.png'
            plt.savefig(write_name)

    if False:
        # global frame_counter 
        frame_counter = frame_counter + 1
        write_name = '../debug_dir/debug_frames/video_frame_'+str(frame_counter)+'.png'
        cv2.imwrite(write_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ## DEBUG - Ends

def process_image(img):
    # process image and generate binary pixel of interests
    preprocessImage, img, img_size = preprocess_image(img, tracker.mtx, tracker.dist)

    # Define properties transformation area
    src, dst, roi_vertices = find_src_dst_roi_vertices(tracker.bot_width, tracker.mid_width, tracker.height_pct, tracker.bottom_trim, img_size)

    # find perspective transform
    warped, Minv, M = find_perspective_transform(src, dst, preprocessImage, img_size)

    # crop image
    warped_cropped = get_roi(warped, roi_vertices)

    # find window centroids
    window_centroids, tracker.recent_centers = find_window_centroids(warped, tracker.window_width, tracker.window_height, tracker.margin, tracker.minpix, tracker.recent_centers, tracker.smooth_factor)

    # find lane mask pixels
    leftx, rightx, l_points, r_points = find_lane_mask_pixels(tracker.window_width, tracker.window_height, warped_cropped, window_centroids)

    # draw the results with both left and right window
    result_raw = draw_results_left_right_window(l_points, r_points, warped_cropped, window_centroids)

    # fit the lane boundaries to the left, right center positions found
    left_lane, right_lane, inner_lane, yvals, res_yvals, left_fitx, right_fitx = find_lane_boundaries(warped_cropped, tracker.window_height, leftx, rightx, tracker.window_width, l_points, r_points)

    # draw the results with final left, right and inner lane lines
    result = draw_results_left_right_inner_lines(img, left_lane, right_lane, inner_lane, Minv, img_size)

    # calculate radius of left & right curvature and the offset of the car on the road
    curverad_left, curverad_right, center_diff, side_pos = calculate_curvature_offset(res_yvals, yvals, leftx, rightx, tracker.ym_per_pix, tracker.xm_per_pix, left_fitx, right_fitx, warped)

    # draw the text showing curvature and offset
    result = draw_curvature_offset(result, curverad_left, curverad_right, center_diff, side_pos)

    ## DEBUG - Begins
    if False:
        # draw details debugging results
        draw_details_debugging_results(src, dst, result, result_raw, warped, warped_cropped, window_centroids, img, M, img_size, s_binary, v_binary, c_binary, gradx, grady, tracker.window_height)
    ## DEBUG - Ends
    
    return result

## DEBUG - Begins
if False:
    images = glob.glob('../debug_dir/debug_frames/temp/video_frame_*.jpg')
## DEBUG - Ends

# Set up the overall class to do all the tracking
tracker = Tracker(Mycalib_width = 9, Mycalib_height = 6, Mywindow_width = 25, Mywindow_height = 80, Mymargin = 50, Myminpix = 50000, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15, Mybot_width = .76, Mymid_width = .08, Myheight_pct = .62, Mybottom_trim = .935)

# set camera calibration
tracker.set_camera_calibration()

# Make a list of test images and save the results
images = glob.glob('../test_images/test*.jpg')
for idx, img in enumerate(images):
    tracker.clear()                     # clear tracking data from previous frames
    result = process_image(img)
    write_name = '../test_images/tracked'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
