import os
import cv2
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import InceptionResnetV1, MTCNN
from datetime import datetime
import pickle
import time

class attendancesystem:
    def __init__(self):
        self.dataset_path = 'attendencesystem'
        self.confidence_threshold = 0.6
        self.model_file = 'model.pkl'
        self.attendance_file = 'attendence.csv'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(keep_all=True, device=self.device, min_face_size=20, thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=True)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.model = None
        self.encoder = None
        self.attendance = set()
        self.person_timers = {}  # Track person detection time
        self.attendance_cooldown = {}  # Track attendance cooldown
        self.face_boxes = {}  # Track stable face boxes
        self.box_history = {}  # Box position history for smoothing
        
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w") as f:
                f.write("Name,Timestamp\n")

    def extract_face_embeddings(self, image):
        embeddings, boxes = [], []
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Enhance image for better far distance detection
            enhanced = cv2.convertScaleAbs(rgb, alpha=1.1, beta=10)
            detected_boxes, probs = self.detector.detect(enhanced)
            
            if detected_boxes is not None and probs is not None:
                for box, prob in zip(detected_boxes, probs):
                    if prob < 0.8:  # Skip very low confidence detections
                        continue
                        
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
                    
                    # Accept smaller faces for far distance
                    if x2 > x1 and y2 > y1 and (x2-x1) * (y2-y1) > 400:  # Min 20x20 pixels
                        face = rgb[y1:y2, x1:x2]
                        # Apply sharpening for small faces
                        if (x2-x1) < 80:  # Small face enhancement
                            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                            face = cv2.filter2D(face, -1, kernel)
                        
                        face = cv2.resize(face, (160, 160))
                        tensor = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                        embedding = self.embedder(tensor.to(self.device)).detach().cpu().numpy().flatten()
                        embeddings.append(embedding)
                        boxes.append((x1, y1, x2, y2))
        except:
            pass
        return embeddings, boxes

    def augment_image(self, img):
        h, w = img.shape[:2]
        center = (w//2, h//2)
        augmented = [
            img,  # Normal
            cv2.flip(img, 1),  # Horizontal flip
            cv2.convertScaleAbs(img, alpha=0.7, beta=0),  # Darker
            cv2.convertScaleAbs(img, alpha=1.3, beta=0),  # Brighter
            cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 5, 1), (w, h)),  # +5°
            cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -5, 1), (w, h)),  # -5°
        ]
        
        # Add scale variations for distance training
        scale_down = cv2.resize(img, None, fx=0.8, fy=0.8)
        scale_down = cv2.resize(scale_down, (w, h))  # Resize back
        augmented.append(scale_down)
        
        # Add noise for robustness
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        augmented.append(noisy)
        
        return augmented

    def load_dataset(self):
        start_time = time.time()
        embeddings, labels = [], []
        
        if not os.path.exists(self.dataset_path):
            return np.array([]), np.array([])
        
        print("Loading dataset with 6-way augmentation...")
        
        for folder in os.listdir(self.dataset_path):
            person_start = time.time()
            person_path = os.path.join(self.dataset_path, folder)
            if not os.path.isdir(person_path):
                continue
                
            count = 0
            for file in os.listdir(person_path):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img = cv2.imread(os.path.join(person_path, file))
                if img is None:
                    continue

                for aug_img in self.augment_image(img):
                    face_embeddings, _ = self.extract_face_embeddings(aug_img)
                    for embedding in face_embeddings:
                        embeddings.append(embedding)
                        labels.append(folder)
                        count += 1
                    
            person_time = time.time() - person_start
            print(f"✓ {folder}: {count} embeddings in {person_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nTotal: {len(embeddings)} embeddings in {total_time:.2f}s")
        return np.array(embeddings), np.array(labels)

    def train_model(self):
        train_start = time.time()
        embeddings, labels = self.load_dataset()
        
        if len(embeddings) == 0:
            return False
        
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(labels)
        
        print(f"Training with {len(embeddings)} samples...")
        
        # Try multiple model configurations for best performance
        best_model = None
        best_score = 0
        
        configs = [
            {'C': 1, 'gamma': 'scale'},
            {'C': 10, 'gamma': 'scale'},
            {'C': 100, 'gamma': 'scale'},
            {'C': 50, 'gamma': 'auto'},
        ]
        
        model_start = time.time()
        for i, config in enumerate(configs):
            print(f"Testing config {i+1}/4: C={config['C']}, gamma={config['gamma']}")
            
            # Train with multiple iterations for stability
            model = SVC(kernel='rbf', probability=True, **config)
            
            # Multiple training iterations
            for iteration in range(3):
                model.fit(embeddings, y_encoded)
            
            # Simple validation score
            train_score = model.score(embeddings, y_encoded)
            print(f"  Score: {train_score:.3f}")
            
            if train_score > best_score:
                best_score = train_score
                best_model = model
        
        self.model = best_model
        model_time = time.time() - model_start
        
        with open(self.model_file, 'wb') as f:
            pickle.dump((self.model, self.encoder), f)
        
        total_time = time.time() - train_start
        print(f"Best score: {best_score:.3f}")
        print(f"Training: {model_time:.2f}s | Total: {total_time:.2f}s")
        return True

    def load_model(self):
        try:
            with open(self.model_file, 'rb') as f:
                self.model, self.encoder = pickle.load(f)[:2]
            return True
        except:
            return self.train_model()

    def mark_attendance(self, name):
        current_time = datetime.now()
        if name in self.attendance_cooldown:
            if (current_time - self.attendance_cooldown[name]).total_seconds() < 5:
                return
        
        if name not in self.attendance:
            self.attendance.add(name)
            with open(self.attendance_file, "a") as f:
                f.write(f"{name},{current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"✓ {name}")
            
        self.attendance_cooldown[name] = current_time

    def smooth_box(self, name, new_box):
        """Smooth box coordinates to prevent blinking"""
        if name not in self.box_history:
            self.box_history[name] = []
        
        self.box_history[name].append(new_box)
        if len(self.box_history[name]) > 3:  # Keep last 3 boxes
            self.box_history[name].pop(0)
        
        # Average the coordinates
        boxes = self.box_history[name]
        x1 = int(sum(box[0] for box in boxes) / len(boxes))
        y1 = int(sum(box[1] for box in boxes) / len(boxes))
        x2 = int(sum(box[2] for box in boxes) / len(boxes))
        y2 = int(sum(box[3] for box in boxes) / len(boxes))
        
        return (x1, y1, x2, y2)

    def recognize_faces(self):
        if not self.model and not self.load_model():
            return
                
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = datetime.now()
            embeddings, boxes = self.extract_face_embeddings(frame)
            detected_people = set()
            current_faces = {}
            
            # Batch process embeddings for efficiency
            if embeddings:
                all_probs = self.model.predict_proba(embeddings)
                
                for i, (embedding, box) in enumerate(zip(embeddings, boxes)):
                    probs = all_probs[i]
                    confidence = np.max(probs)
                    
                    if confidence >= self.confidence_threshold:
                        name = self.encoder.inverse_transform([np.argmax(probs)])[0]
                        detected_people.add(name)
                        
                        # Smooth the box coordinates
                        smooth_box = self.smooth_box(name, box)
                        current_faces[name] = {'box': smooth_box, 'confidence': confidence}
                        
                        if name not in self.person_timers:
                            self.person_timers[name] = current_time
            
            # Draw all detected faces with stable boxes
            for name, face_data in current_faces.items():
                x1, y1, x2, y2 = face_data['box']
                confidence = face_data['confidence']
                
                time_diff = (current_time - self.person_timers[name]).total_seconds()
                
                if time_diff >= 3:  # 3 second detection
                    self.mark_attendance(name)
                    del self.person_timers[name]  # Remove timer after attendance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"{name} - MARKED", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"{time_diff:.1f}/3.0s", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw unknown faces
            if embeddings:
                all_probs = self.model.predict_proba(embeddings)
                for i, (embedding, box) in enumerate(zip(embeddings, boxes)):
                    probs = all_probs[i]
                    confidence = np.max(probs)
                    
                    if confidence < self.confidence_threshold:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Clean up timers and box history for people no longer detected
            for person in list(self.person_timers.keys()):
                if person not in detected_people:
                    del self.person_timers[person]
                    if person in self.box_history:
                        del self.box_history[person]

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = attendancesystem()
    system.recognize_faces()