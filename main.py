import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(title, image):
    plt.figure(figsize=(4, 4))
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def apply_kernel(image, kernel):
    return cv2.filter2D(image, -1, kernel)


img = cv2.imread("data/kitty.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

show_image("Original Grayscale", gray)

kernels = {
    "Blur": np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]),
    "Horizontal Edges": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]),
    "Vertical Edges": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    "Fuzzy": np.array([
        [1/3, 1/5, 1/3],
        [1/5, -19/15, 1/5],
        [1/3, 1/5, 1/3]
    ]),
    # Weird fuzzy effect, hard to describe, but it creates a unique texture.

    "High Contrasting": np.array([
        [-1, 1, -1],
        [-1, 8, -1],
        [-1, 1, -1]
    ]),
    # Makes it brighter and more contrasted.

    "Cross": np.array([
        [1, -1, 1],
        [-1, 1, -1],
        [1, -1, 1]
    ])
    # Creates a cross pattern effect. Looks weird too.
}

for name, kernel in kernels.items():
    result = apply_kernel(gray, kernel)
    show_image(name, result)

# STUDENT TASK:
# 1. Add two more kernels to the dictionary.
# 2. Show the results.
# 3. Write a short explanation of what each kernel does.