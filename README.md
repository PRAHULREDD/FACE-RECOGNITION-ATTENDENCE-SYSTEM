# 🧠 Face Recognition Attendance System

This is a Python-based project that **automatically marks attendance using face recognition** through a webcam. It uses AI and computer vision libraries like OpenCV and FaceNet.

---

## 🚀 Features (What It Can Do)
- 👥 **Detects Multiple Faces** at the same time.
- 🧪 **Checks Model Quality** automatically (overfitting or underfitting).
- 🧠 **Chooses Best Model Settings** to improve accuracy.
- 📊 **Live Accuracy Display** on screen.
- ⚙️ **Easy Configuration** using a JSON file.

---

## 🛠️ Setup Instructions (For Beginners)

1. 📦 **Install Required Libraries**  
   In your terminal, type:
   ```
   pip install -r requirements.txt
   ```

2. 🗂️ **Create a Dataset Folder**  
   Make a folder named:
   ```
   attendencesystem/
   ```

3. 👤 **Add Face Images for Each Person**  
   Inside `attendencesystem/`, create one folder per person.  
   Each folder should have **multiple clear face images** of that person.  
   Example:
   ```
   attendencesystem/Akash/
   attendencesystem/Sneha/
   ```

---

## ▶️ How to Use It

> **Main File to Run the System:**  
> ✅ `main.py` is the main program you need to run.  

| Task | Command |
|------|---------|
| ✅ **Start Face Recognition System** | `python main.py` |
| 🧪 Validate Model Accuracy | `python validate_model.py` |
| 📄 View Attendance Records | `python utils.py view` |
| 🧹 Clear Attendance Records | `python utils.py clear` |

---

## 🎮 In-Program Controls

- Press `q` → Quit the program  
- Press `r` → Retrain the model using current dataset  

---

## 📈 Model Behavior

The system will:
- Warn you if it detects **overfitting** (great results in training, poor in testing)
- Warn about **underfitting** (low accuracy on test data)
- Pick the **best model settings**
- Save model performance details automatically

---

## ⚙️ Configuration File Explained (`config.json`)

You can control some settings using the config file:

| Setting | What It Means |
|---------|----------------|
| `confidence_threshold` | How confident the system should be (0.0–1.0) to mark someone present |
| `dataset_path` | Location of your image folders |
| `model_file` | Filename where the trained model is saved |
| `attendance_file` | CSV file where attendance will be recorded |
| `min_face_size` | Minimum size of face (in pixels) to detect |

---
