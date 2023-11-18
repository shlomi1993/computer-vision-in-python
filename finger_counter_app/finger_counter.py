import cv2
import numpy as np

from dataclasses import dataclass
from functools import cached_property
from sklearn.metrics import pairwise


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


@dataclass
class ROI:
    top: int
    bottom: int
    right: int
    left: int


class FingerCounter:
    WINDOW_NAME = 'Finger Count'
    
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.background = None
        self.roi = ROI(
            top=int(self.frame_height / 2 - 150),
            bottom=int(self.frame_height / 2 + 150),
            right=int(self.frame_width / 2 - 150),
            left=int(self.frame_width / 2 + 150)
        )
        self.accumulated_weight = 0.5  # Start with a halfway point between 0 and 1 of accumulated weight
        self.is_running = False

    @cached_property
    def frame_height(self):
        return int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @cached_property
    def frame_width(self):
        return int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    def stop(self):
        self.is_running = False
        self.camera.release()
        cv2.destroyAllWindows()
        
    def calc_accum_avg(self, frame, accumulated_weight):
        if self.background is None:
            self.background = frame.copy().astype('float')
        else:
            cv2.accumulateWeighted(frame, self.background, accumulated_weight)
        
    def segment_hand(self, frame, threshold=25):
        diff = cv2.absdiff(self.background.astype('uint8'), frame)
        thresholded_frame = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        
        contours = cv2.findContours(thresholded_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours) == 0:
            return None

        hand_segment = max(contours, key=cv2.contourArea)  # The largest external contour should be the hand

        return thresholded_frame, hand_segment

    @staticmethod
    def count_fingers(thresholded, hand_segment):
        
        # Calculate the convex hull of the hand segment - it will have at least 4 most outward points
        conv_hull = cv2.convexHull(hand_segment)
        
        # Find the top, bottom, left, and right most outward points and turn them into tuples
        top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
        bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
        left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
        right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

        # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
        cX = (left[0] + right[0]) // 2
        cY = (top[1] + bottom[1]) // 2

        # Calculate the largest Euclidean distance between the center of the hand and the left, right, top, and bottom.
        max_distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0].max()
                
        # Create a circle with 90% radius of the max euclidean distance
        radius = int(0.8 * max_distance)
        circumference = (2 * np.pi * radius)

        # Now grab an ROI of only that circle
        circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
        
        # Draw the circular ROI
        cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
        
        # Using bit-wise AND with the cirle ROI as a mask.
        # This then returns the cut out obtained using the mask on the thresholded hand image.
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

        # Grab contours in circle ROI
        contours = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        # Loop through the contours to count any fingers
        count = 0
        for cnt in contours:            
            (x, y, w, h) = cv2.boundingRect(cnt)
            
            # Condition 1: Contour region is not the very bottom of hand area (the wrist)
            out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
            
            # Condition 2: Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
            limit_points = ((circumference * 0.25) > cnt.shape[0])
            
            if  out_of_wrist and limit_points:
                count += 1

        return count

    def run(self):
        n_frames = 0
        self.is_running = True
        while self.is_running:

            # get the current frame
            ret, frame = self.camera.read()

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            frame_copy = frame.copy()

            # Grab the ROI from the frame
            roi = frame[self.roi.top:self.roi.bottom, self.roi.right:self.roi.left]

            # Apply grayscale and blur to ROI
            roi_grayscaled = cv2.cvtColor(src=roi, code=cv2.COLOR_BGR2GRAY)
            roi_grayscaled_blurred = cv2.GaussianBlur(src=roi_grayscaled, ksize=(7, 7), sigmaX=0)

            # Calculate the average of the background for the first 60 frames
            if n_frames < 60:
                self.calc_accum_avg(roi_grayscaled_blurred, self.accumulated_weight)
                if n_frames <= 59:
                    org = (self.roi.right, self.roi.bottom + 35) 
                    cv2.putText(frame_copy, 'Initializing...', org, cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
                    cv2.imshow(self.WINDOW_NAME, frame_copy)
                    
            else:  # Here we can assume we have the background so we can segment the hand                
                hand = self.segment_hand(roi_grayscaled_blurred)
                if hand:
                    thresholded_frame, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.roi.right, self.roi.top)], -1, BLUE, 1)
                    fingers = self.count_fingers(thresholded_frame, hand_segment)
                    cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
                    cv2.imshow('Thesholded frame', thresholded_frame)

            # Draw ROI Rectangle on frame copy
            cv2.rectangle(frame_copy, (self.roi.left, self.roi.top), (self.roi.right, self.roi.bottom), GREEN, 5)

            # increment the number of frames for tracking
            n_frames += 1

            # Display the frame with segmented hand
            cv2.imshow(self.WINDOW_NAME, frame_copy)

            # Close windows with Esc
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop()


def main():
    FingerCounter().run()


if __name__ == '__main__':
    main()
