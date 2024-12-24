#!/opt/venv/bin/python

import cv2
import numpy as np

def predict_frame(reference_frame, motion_vectors, block_size):
    """Generate a predicted frame using the reference frame and motion vectors."""
    height, width = reference_frame.shape[:2]
    predicted_frame = np.zeros_like(reference_frame)

    for (x, y, dx, dy) in motion_vectors:
        # Compute the destination position in the reference frame
        x_dest = x + dx
        y_dest = y + dy

        # Validate block boundaries
        if (0 <= x < width - block_size + 1) and (0 <= y < height - block_size + 1) and \
           (0 <= x_dest < width - block_size + 1) and (0 <= y_dest < height - block_size + 1):

            # Copy the block from the reference frame to the predicted frame
            predicted_frame[y:y + block_size, x:x + block_size] = \
                reference_frame[y_dest:y_dest + block_size, x_dest:x_dest + block_size]

    return predicted_frame
