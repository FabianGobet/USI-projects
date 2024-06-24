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


def main():
    image = cv2.imread("./data/WSC sample.png")
    # perfect grid to try distortion
    # image = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Uncomment to see the distortion effect on a perfect grid i.e. see lines above scoreboard
    # for i in range(0, 1280, 40):
    #     image = cv2.line(image, (i, 0), (i, 720), (255, 255, 255), 1)
    # for i in range(0, 720, 40):
    #     image = cv2.line(image, (0, i), (1280, i), (255, 255, 255), 1)

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
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs_pin_cushion)

    undistorted_image = undistort_image(
        undistorted_image, camera_matrix, dist_coeffs_barrel
    )

    clean_image = undistorted_image.copy()

    # Lines to draw
    lines = [
        [360, 41],  # Top-left corner
        [919, 41],  # Top-right corner
        [1055, 620],  # Bottom-right corner
        [220, 620],  # Bottom-left corner
    ]
    for point in lines:
        undistorted_image = cv2.circle(
            undistorted_image, (int(point[0]), int(point[1])), 3, (255, 255, 255), -1
        )

    # Line from top-left to top-right
    undistorted_image2 = cv2.line(
        undistorted_image,
        (lines[0][0], lines[0][1]),
        (lines[1][0], lines[1][1]),
        (255, 255, 255),
        1,
    )

    # Line from top-right to bottom-right
    undistorted_image2 = cv2.line(
        undistorted_image2,
        (lines[1][0], lines[1][1]),
        (lines[2][0], lines[2][1]),
        (255, 255, 255),
        1,
    )

    # Line from bottom-right to bottom-left
    undistorted_image2 = cv2.line(
        undistorted_image2,
        (lines[2][0], lines[2][1]),
        (lines[3][0], lines[3][1]),
        (255, 255, 255),
        1,
    )

    # Line from bottom-left to top-left
    undistorted_image2 = cv2.line(
        undistorted_image2,
        (lines[3][0], lines[3][1]),
        (lines[0][0], lines[0][1]),
        (255, 255, 255),
        1,
    )

    image = cv2.line(image, (360, 41), (919, 41), (255, 255, 255), 1)
    image = cv2.line(image, (919, 41), (1055, 624), (255, 255, 255), 1)
    image = cv2.line(image, (1055, 624), (220, 624), (255, 255, 255), 1)
    image = cv2.line(image, (220, 624), (360, 41), (255, 255, 255), 1)

    # image = cv2.line(image, (0, 720 - 98), (1280, 720 - 98), (255, 255, 255), 1)

    plt.imshow(undistorted_image2)
    plt.title("Undistorted Image with Corner Points")
    # plt.show()

    cv2.imwrite("undistorted_image.jpg", undistorted_image2)
    cv2.imwrite("distorted_image.jpg", image)
    cv2.imwrite("clean_image.jpg", clean_image)


if __name__ == "__main__":
    main()
