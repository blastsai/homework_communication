#!/opt/venv/bin/python

import cv2
import numpy as np

def predict_frame_zero_padded(reference_frame, motion_vectors, block_size):
    """Generate a predicted frame using the reference frame and motion vectors."""
    height, width = reference_frame.shape[:2]
    predicted_frame = np.zeros_like(reference_frame)

    for (x, y, dx, dy) in motion_vectors:
        # Compute the destination position in the reference frame
        x_dest = min(max(x + dx, 0), width - block_size)
        y_dest = min(max(y + dy, 0), height - block_size)

        # Validate block boundaries
        if (0 <= x < width - block_size + 1) and (0 <= y < height - block_size + 1):

            # Extract the predicted block
            predicted_block = reference_frame[y_dest:y_dest + block_size, x_dest:x_dest + block_size]

            # Pad the predicted block if necessary
            predicted_block_padded = np.zeros((block_size, block_size), dtype=np.uint8)
            predicted_block_padded[:predicted_block.shape[0], :predicted_block.shape[1]] = predicted_block

            # Place the padded block into the predicted frame
            x_end = min(x + block_size, width)
            y_end = min(y + block_size, height)
            predicted_frame[y:y_end, x:x_end] = predicted_block_padded[:y_end - y, :x_end - x]

    return predicted_frame
