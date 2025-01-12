import os
import cv2
import numpy as np
import mediapipe as mp

dataset_path = "dataset"

# Veri seti için gerekli klasörleri oluşturma
def create_dataset_folders():
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Veri klasörlerini oluşturma
    classes = [chr(i) for i in range(65, 91)]
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
    
    print("All folders created.")

# MediaPipe el tespit modelini başlatma
def initialize_hand_detection():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

# El landmarkları için sınırlayıcı kareyi hesaplama
def get_hand_boundaries(hand_landmarks, frame_shape):
    height, width = frame_shape[:2]
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    points = []

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        points.append((x, y))
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)

    return points, (x_min, y_min, x_max, y_max)

# El için maske oluşturma ve işleme
def create_hand_mask(points, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    # Avuç içini doldurma
    palm_points = np.array([points[0], points[1], points[5], points[17]])
    cv2.fillPoly(mask, [palm_points], 255)
    
    # Avuç içi çizgilerini kalınlaştırma
    cv2.line(mask, points[0], points[1], 255, thickness=50)
    cv2.line(mask, points[1], points[5], 255, thickness=50)
    cv2.line(mask, points[5], points[17], 255, thickness=50)
    cv2.line(mask, points[17], points[0], 255, thickness=50)
    
    # Parmak çizgileri
    for i in range(len(points)-1):
        if i % 4 != 0:
            cv2.line(mask, points[i], points[i + 1], 255, thickness=20)
    
    # Maskeyi genişletme
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)
    dilated_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
    
    return dilated_mask

# Eli çevreleyen kare sınırları hesaplama
def get_square_boundaries(boundaries, frame_shape):
    x_min, y_min, x_max, y_max = boundaries
    height, width = frame_shape[:2]
    
    # El bölgesinin merkezi
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # El bölgesinin genişliği ve yüksekliği
    width_hand = x_max - x_min
    height_hand = y_max - y_min
    
    # Padding ekleme ve kare boyutunu belirleme
    square_size = int(max(width_hand, height_hand) + 100)
    
    # Yeni sınırları hesaplama
    x_min = center_x - square_size // 2
    y_min = center_y - square_size // 2
    x_max = center_x + square_size // 2
    y_max = center_y + square_size // 2
    
    # Sınırların frame içinde kalmasını sağlama
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > width:
        x_min -= (x_max - width)
        x_max = width
    if y_max > height:
        y_min -= (y_max - height)
        y_max = height
    
    return x_min, y_min, x_max, y_max

def main():
    create_dataset_folders()
    hands = initialize_hand_detection()
    cap = cv2.VideoCapture(0)
    
    # Sınıf seçimi
    current_class = input("Enter the class: ")
    class_path = os.path.join(dataset_path, current_class)
    
    if not os.path.exists(class_path):
        print(f"Folder {current_class} not found.")
        cap.release()
        return
    
    print(f"Saving images for {current_class}. Press \"s\" to take a photo.")
    image_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Frame not available.")
            break
        
        frame = cv2.flip(frame, 1)
        result = np.zeros_like(frame)
        display_frame = frame.copy() # Görüntüleme için kopya oluşturma
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points, boundaries = get_hand_boundaries(hand_landmarks, frame.shape)
                mask = create_hand_mask(points, frame.shape)
                square_bounds = get_square_boundaries(boundaries, frame.shape)
                x_min, y_min, x_max, y_max = square_bounds
                
                # Kareyi görüntüleme frame'ine çizme
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # El maskesi ile görüntüyü birleştirme
                result = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Kare içindeki görüntüyü alma
                hand_square = result[y_min:y_max, x_min:x_max]
                if hand_square.size > 0:
                    # Görüntüyü kaydetme
                    if cv2.waitKey(1) & 0xFF == ord("s"):
                        # Kareyi eşit boyutlara getirme
                        square_size = max(hand_square.shape[0], hand_square.shape[1])
                        square_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
                        
                        y_offset = (square_size - hand_square.shape[0]) // 2
                        x_offset = (square_size - hand_square.shape[1]) // 2
                        
                        square_img[y_offset:y_offset+hand_square.shape[0], 
                                 x_offset:x_offset+hand_square.shape[1]] = hand_square
                        
                        # Son boyuta yeniden boyutlandırma
                        final_img = cv2.resize(square_img, (64, 64))
                        
                        image_name = f"{image_count + 1}.jpg"
                        image_path = os.path.join(class_path, image_name)
                        cv2.imwrite(image_path, final_img)
                        print(f"{image_name} saved.")
                        image_count += 1
        
        # Hem el maskesi hem de kare sınırlarını göster
        combined_view = cv2.addWeighted(display_frame, 0.1, result, 0.9, 0)
        cv2.imshow("Data Collection", combined_view)
        #cv2.imshow("Data Collection", result)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    print(f"\n{image_count} images saved for {current_class}.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()