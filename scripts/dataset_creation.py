import os
import cv2

def main():
    path_to_images = "data"
    folders = ["train", "valid", "test"]

    for folder in folders:
        folder_path = os.path.join(path_to_images, folder)
        source_path = os.path.join(folder_path, "source")
        target_path = os.path.join(folder_path, "target")
        image_names = os.listdir(source_path)

        for image_name in image_names:
            img = cv2.imread(os.path.join(source_path, image_name), cv2.IMREAD_GRAYSCALE)

            edges = cv2.Canny(img,100,200)

            cv2.imwrite(os.path.join(target_path, image_name), edges)


if __name__ == "__main__":
    main()