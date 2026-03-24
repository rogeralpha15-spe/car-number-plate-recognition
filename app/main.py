import cv2
from ocr_utils import read_plate_text

image_path = "sample_images/image_1.png"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

x, y, w, h = 20, 20, 300, 120
plate = img[y:y+h, x:x+w]

text, processed = read_plate_text(plate)

print("Detected Number Plate Text:", text)

cv2.imwrite("plate_crop.jpg", plate)
cv2.imwrite("plate_processed.jpg", processed)
print("Saved plate_crop.jpg and plate_processed.jpg")
