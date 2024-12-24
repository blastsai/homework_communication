#!/opt/venv/bin/python

import cv2
import numpy as np
import time
from predict_frame_zero_padded import predict_frame_zero_padded
from predict_frame import predict_frame

def draw_motion_vectors(frame, motion_vectors):
    """Visualize motion vectors on a given frame."""
    for (x, y, dx, dy) in motion_vectors:
        start_point = (x, y)
        end_point = (x + dx, y + dy)
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 1, tipLength=0.2)
    return frame

def full_search(reference_frame, target_frame, block_size, search_range):
    """Perform exhaustive motion estimation (Full Search) between frames."""
    height, width = reference_frame.shape[:2]
    motion_vectors = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            curr_block = target_frame[y:y + block_size, x:x + block_size]
            min_cost = float('inf')
            best_dx, best_dy = 0, 0

            # Search within the defined range
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_x = x + dx
                    ref_y = y + dy

                    if (0 <= ref_x < width - block_size + 1) and (0 <= ref_y < height - block_size + 1):
                        ref_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        cost = np.sum(np.abs(ref_block - curr_block))

                        if cost < min_cost:
                            min_cost = cost
                            best_dx, best_dy = dx, dy

            motion_vectors.append((x, y, best_dx, best_dy))

    return motion_vectors

def main():
    """Main function to execute motion estimation and display results."""
    # Initialize camera feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Capture two sequential frames
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Failed to capture the first frame.")
        return

    time.sleep(1)

    ret, frame2 = cap.read()
    if not ret:
        print("Error: Failed to capture the second frame.")
        return

    # Save original frames for debugging
    cv2.imwrite("frame1.jpg", frame1)
    cv2.imwrite("frame2.jpg", frame2)

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Set parameters for motion estimation
    block_size = 16
    search_range = 8

    # Calculate motion vectors using Full Search
    motion_vectors = full_search(gray1, gray2, block_size, search_range)

    # Generate predicted frame and residual
    predicted_frame = predict_frame_zero_padded(gray1, motion_vectors, block_size)
    residual = gray2 - predicted_frame

    # Save intermediate results
    cv2.imwrite("residual.jpg", residual)
    cv2.imwrite("predicted_frame.jpg", predicted_frame)

    # Reconstruct the second frame
    reconstructed_frame = predicted_frame + residual
    cv2.imwrite("reconstructed_frame.jpg", reconstructed_frame)

    # Overlay motion vectors on the first frame
    motion_frame = draw_motion_vectors(frame1.copy(), motion_vectors)

    # Display all results
    cv2.imshow("Original Frame 1", frame1)
    cv2.imshow("Original Frame 2", frame2)
    cv2.imshow("Motion Vectors", motion_frame)
    cv2.imshow("Predicted Frame", predicted_frame)
    cv2.imshow("Residual", residual)
    cv2.imshow("Reconstructed Frame", reconstructed_frame)

    # Wait for user interaction
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
