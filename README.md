# Face Recognition Attendance System

## Features
- **Multiple Face Detection**: Detects and recognizes multiple faces simultaneously
- **Model Validation**: Automatic overfitting/underfitting detection
- **Smart Model Selection**: Tests different configurations to find the best model
- **Performance Monitoring**: Real-time accuracy display
- **Configurable Settings**: JSON-based configuration

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create dataset folder: `attendencesystem/`
3. Add person folders: `attendencesystem/PersonName/` with images

## Usage
- **Run system**: `python main.py`
- **Validate model**: `python validate_model.py`
- **View attendance**: `python utils.py view`
- **Clear attendance**: `python utils.py clear`

## Controls
- `q`: Quit
- `r`: Retrain model with validation

## Model Performance
The system automatically:
- Detects overfitting (train accuracy >> test accuracy)
- Detects underfitting (low test accuracy < 0.8)
- Selects optimal model configuration
- Saves performance metrics with the model

## Configuration (`config.json`)
- `confidence_threshold`: Recognition confidence (0.0-1.0)
- `dataset_path`: Path to training images
- `model_file`: Saved model filename
- `attendance_file`: Attendance CSV filename
- `min_face_size`: Minimum face detection size