import cv2
import argparse
import numpy as np
import json

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="Draw a polygon on an image and get coordinates.")
    parser.add_argument("image_path", help="Path to the image or video frame")
    args = parser.parse_args()

    # Try as video first
    cap = cv2.VideoCapture(args.image_path)
    if cap.isOpened():
        ret, img = cap.read()
        cap.release()
        if not ret:
            print(f"Error: Could not read frame from video {args.image_path}")
            return
    else:
        # Fallback to image
        img = cv2.imread(args.image_path)
        
    if img is None:
        print(f"Error: Could not load image or video at {args.image_path}")
        return

    window_name = "Draw Polygon (Click to add points, 'c' to clear, 'q' or ESC to quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_img = img.copy()
        
        # Draw the points and lines
        if len(points) > 0:
            for p in points:
                cv2.circle(display_img, p, 5, (0, 0, 255), -1)
            
            if len(points) > 1:
                cv2.polylines(display_img, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
                
        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break
        elif key == ord('c'):
            points.clear()
            print("Points cleared")

    cv2.destroyAllWindows()
    
    print("\n--- Final Polygon Points ---")
    print(json.dumps(points))
    print("----------------------------\n")

if __name__ == "__main__":
    main()
