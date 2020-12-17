import cv2

png = cv2.imread(r"C:\\Users\\ASUS\\Desktop\\AMLS_Assignment\\AMLS_PROJECT\\Datasets\\cartoon_set\\img\\0.png", cv2.IMREAD_COLOR)
jpg = cv2.imread(r"C:\\Users\\ASUS\\Desktop\\AMLS_Assignment\\AMLS_PROJECT\\Datasets\\cartoon_set\\img_jpg\\0.jpg", cv2.IMREAD_COLOR)

print(png)
print(jpg)
cv2.imshow('png', png)
cv2.imshow('jpg', jpg)