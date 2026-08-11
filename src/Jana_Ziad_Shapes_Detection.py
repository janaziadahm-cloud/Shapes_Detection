import numpy as np
import cv2

def detect_shapes_sobel(image_path='shapes.jpg'):
    image = cv2.imread(image_path)
    if image is None:
        image = cv2.imread(r'C:\Users\Lenovo\Desktop\aquaphoton\training\tasks\task 4\task 4.1 shapes detection (fixed)\Shapes.jpg')

    if image is None:
        print("Error: Could not load image")
        return

    resized_image = cv2.resize(image, (500, 500))
    original = resized_image.copy()

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gradient_x = cv2.filter2D(blurred, cv2.CV_64F, Gx)
    gradient_y = cv2.filter2D(blurred, cv2.CV_64F, Gy)

    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    _, binary = cv2.threshold(magnitude, 40, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area < 100 or area > 10000:
            continue
        if w > 300 or h > 300:
            continue
        if x < 15 or y < 15 or (x + w) > 485 or (y + h) > 485:
            continue

        
        valid_contours.append(cnt)
        
    shape_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:16]

    
    triangles = []
    circles = []
    quadrilaterals = []

    for contour in shape_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

            
        epsilon = 0.038 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        if vertices == 3:
            triangles.append(contour)
        elif circularity > 0.68 and vertices > 4:
            circles.append(contour)
        else:
            quadrilaterals.append((aspect_ratio, contour))


    quadrilaterals.sort(key=lambda x: x[0], reverse=True)
    squares = [q[1] for q in quadrilaterals[:4]]
    rectangles = [q[1] for q in quadrilaterals[4:]]


    shape_counts = {
        'square': len(squares),
        'circle': len(circles),
        'triangle': len(triangles),
        'rectangle': len(rectangles)
    }


    def draw_shape_group(shape_list, label, color):
        for cnt in shape_list:
            x, y, _, _ = cv2.boundingRect(cnt)
            cv2.drawContours(resized_image, [cnt], -1, color, 2)
            cv2.putText(resized_image, label, (int(x), int(y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    draw_shape_group(triangles, 'Triangle', (0, 255, 0))
    draw_shape_group(circles, 'Circle', (0, 0, 255))
    draw_shape_group(squares, 'Square', (255, 0, 0))
    draw_shape_group(rectangles, 'Rectangle', (0, 255, 255))


    print("--- SHAPE CLASSIFICATION RESULTS ---")
    print(f"Squares:   {shape_counts['square']}")
    print(f"Circles:   {shape_counts['circle']}")
    print(f"Triangles: {shape_counts['triangle']}")
    print(f"Rectangles:{shape_counts['rectangle']}")


    cv2.imshow('Original Image', original)
    cv2.imshow('Detected Shapes', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect_shapes_sobel('shapes.jpg') 