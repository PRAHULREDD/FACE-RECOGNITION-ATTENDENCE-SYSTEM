# Face Recognition Attendance System

This project is a real-time face recognition attendance system using deep learning and classical machine learning techniques. It detects faces from a webcam feed, recognizes them using FaceNet embeddings and an SVM classifier, and marks attendance after verifying presence for a defined duration.

## 🚀 Features

- Real-time face detection and recognition via webcam
- MTCNN for face detection with small-face enhancement
- FaceNet (InceptionResnetV1) for 512D embedding generation
- SVM classifier trained on augmented face embeddings
- 6+ types of data augmentation for better generalization
- Box smoothing to reduce flicker and false triggers
- Attendance marking after stable 3-second detection
- CSV-based attendance logging with timestamps
- Cooldown mechanism to avoid duplicate entries

## 🧰 Tech Stack

- Python
- OpenCV
- PyTorch + facenet-pytorch
- scikit-learn (SVM, LabelEncoder)
- NumPy
- MTCNN + InceptionResnetV1

## 📂 Folder Structure

```
attendencesystem/
├── 12345_Rahul/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── 67890_Akash/
│   ├── img1.jpg
│   └── ...
```

## 📁 Files Generated

- `model.pkl`: Trained SVM model and LabelEncoder
- `attendence.csv`: Attendance log file
- `main.py`: Main recognition script

## 🧪 How It Works

1. Load training dataset with 6+ augmentations
2. Extract embeddings using FaceNet
3. Train SVM with best configuration
4. Use webcam to recognize faces in real-time
5. If person is stable for 3 seconds, mark attendance
6. Save timestamped entry in `attendence.csv`

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

## ✍️ Attendance Log Example

```
Name,Timestamp
12345_Rahul,2025-08-08 10:32:11
67890_Akash,2025-08-08 10:35:07
```

## 📌 Notes

- Ensure the dataset folder structure follows the format above.
- Press `q` to quit the webcam window.
- Model will auto-train if `model.pkl` is missing.

## 👨‍💻 Author

P. Rahul Reddy  
[GitHub](https://github.com/PRAHULREDD) | [LinkedIn](https://www.linkedin.com/in/rahul-reddy-7bb9a9324/)
