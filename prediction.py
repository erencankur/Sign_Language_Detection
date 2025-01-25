import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model_path = "model.keras"

def initialize_hand_detection():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

def load_prediction_model():
    return load_model(model_path)

def get_class_mapping():
    return [chr(i) for i in range(65, 91)]

def process_hand_landmarks(hand_landmarks, frame_shape):
    height, width = frame_shape[:2]
    points = []
    x_min, y_min = width, height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        points.append((x, y))
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)

    return points, (x_min, y_min, x_max, y_max)

def create_hand_mask(points, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    palm_points = np.array([points[0], points[1], points[5], points[17]])
    cv2.fillPoly(mask, [palm_points], 255)
    
    cv2.line(mask, points[0], points[1], 255, thickness=50)
    cv2.line(mask, points[1], points[5], 255, thickness=50)
    cv2.line(mask, points[5], points[17], 255, thickness=50)
    cv2.line(mask, points[17], points[0], 255, thickness=50)
    
    for i in range(len(points)-1):
        if i % 4 != 0:
            cv2.line(mask, points[i], points[i + 1], 255, thickness=20)
    
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)
    dilated_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
    
    return dilated_mask

def get_square_boundaries(boundaries, frame_shape):
    x_min, y_min, x_max, y_max = boundaries
    height, width = frame_shape[:2]
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    width_hand = x_max - x_min
    height_hand = y_max - y_min
    
    square_size = int(max(width_hand, height_hand) + 100)
    
    # Yeni sınırları hesaplama
    new_x_min = center_x - square_size // 2
    new_y_min = center_y - square_size // 2
    new_x_max = center_x + square_size // 2
    new_y_max = center_y + square_size // 2
    
    if new_x_min < 0:
        new_x_max -= new_x_min
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min
        new_y_min = 0
    if new_x_max > width:
        new_x_min -= (new_x_max - width)
        new_x_max = width
    if new_y_max > height:
        new_y_min -= (new_y_max - height)
        new_y_max = height
    
    return new_x_min, new_y_min, new_x_max, new_y_max

def predict_hand_sign(model, hand_square, class_mapping):
    hand_square = cv2.resize(hand_square, (64, 64))
    hand_square = hand_square / 255.0
    hand_square = np.expand_dims(hand_square, axis=0)

    predictions = model.predict(hand_square)
    predicted_class = np.argmax(predictions)
    return class_mapping[predicted_class]

def main():
    hands = initialize_hand_detection()
    model = load_prediction_model()
    class_mapping = get_class_mapping()
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            print("Frame not available.")
            break

        frame = cv2.flip(frame, 1)
        result = np.zeros_like(frame)
        display_frame = frame.copy()
        predicted_character = "DNE"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points, boundaries = process_hand_landmarks(hand_landmarks, frame.shape)
                mask = create_hand_mask(points, frame.shape)
                square_bounds = get_square_boundaries(boundaries, frame.shape)
                x_min, y_min, x_max, y_max = square_bounds

                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                result = cv2.bitwise_and(frame, frame, mask=mask)

                hand_square = result[y_min:y_max, x_min:x_max]
                if hand_square.size > 0:
                    square_size = max(hand_square.shape[0], hand_square.shape[1])
                    square_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
                    
                    y_offset = (square_size - hand_square.shape[0]) // 2
                    x_offset = (square_size - hand_square.shape[1]) // 2
                    
                    square_img[y_offset:y_offset+hand_square.shape[0], 
                             x_offset:x_offset+hand_square.shape[1]] = hand_square
                    
                    predicted_character = predict_hand_sign(model, square_img, class_mapping)

        cv2.putText(frame, f"Predicted: {predicted_character}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()