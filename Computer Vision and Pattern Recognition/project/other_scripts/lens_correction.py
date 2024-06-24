import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    Lens Distortion Correction for the WSC Pool Table Footage.

    The function undistort_image takes an image, camera matrix, and distortion coefficients as input and returns an undistorted image.

    The camera matrix is a 3x3 matrix that contains the intrinsic parameters of the camera, such as the focal length and principal point.
    Since we don't have the actual camera parameters, we can use an estimated camera matrix based on the resolution of the image.

    The distortion coefficients are a 1x4 or 1x5 vector that contains the distortion coefficients of the camera.
    There are two types of distortion: radial distortion and tangential distortion.

    This is the demo for the lens distortion correction for the WSC pool table footage.

"""


def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted_image = cv2.undistort(
        image, camera_matrix, dist_coeffs, None, new_camera_matrix
    )
    return undistorted_image


def lens_correction(image, view_output=False, save_image=False, black_edges=False):
    camera_matrix = np.array(
        [[1280, 0, 640], [0, 720, 360], [0, 0, 1]], dtype=np.float32
    )

    dist_coeffs_pin_cushion = np.array(
        [0.1, 0.1, 0.0, 0.0], dtype=np.float32
    )  # corrects top and bottom edges
    dist_coeffs_barrel = np.array(
        [-0.02, -0.02, 0.0, 0.0], dtype=np.float32
    )  # corrects left and right edges

    # Undistort the image
    corrected_image = undistort_image(image, camera_matrix, dist_coeffs_pin_cushion) # Apply pin cushion distortion correction (top and bottom edges)

    corrected_image = undistort_image(
        corrected_image, camera_matrix, dist_coeffs_barrel
    ) # Apply barrel distortion correction (left and right edges)

    if view_output:
        # Correct BGRA to RGB
        output_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
        plt.imshow(output_image)
        plt.title("Undistorted Image with Corner Points")
        plt.show()

    # Zoom in to remove the black edges
    if black_edges:
        corrected_image = corrected_image[20:-20, 20:-20]

    if save_image:
        cv2.imwrite("corrected_image.png", corrected_image)

    return corrected_image


if __name__ == "__main__":
    image = cv2.imread("./data/WSC sample.png")
    corrected_image = lens_correction(image, view_output=False, save_image=True, black_edges=False)
