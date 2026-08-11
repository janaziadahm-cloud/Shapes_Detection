**Task 4.1: Shapes Detection**  
\- this project detects circles, squares, rectangles and triangles   
\- using custom Sobel edge detection and tilt-corrected contour analysis while filtering out outer page frames.

\- how it works:

* preprocess: Grayscale conversion and 5x5 Gaussian Blur for noise removal.  
* sobel Gradients: Calculates Gₓ and Gᵧ using cv2.filter2D.  
* binary Edge Map: Thresholds gradient magnitude √(Gx² \+ Gy²) and seals boundaries with cv2.MORPH\_CLOSE.  
* filter Contours: Removes outer page borders/noise to isolate the grid shapes.


\- classify:

* triangles: 3 vertices via approxPolyDP.  
* circles: circularity \> 0.68 and \> 4 vertices.  
* squares & rectangles: computes tilt-corrected aspect ratios with minAreaRect, highest 4 ratios are squares, lowest 4 are rectangles.


\- prints shape counts to the terminal.  
\- displays annotated image with color-coded labels   
(Green: Triangle, Red: Circle, Blue: Square, Yellow: Rectangle).  
