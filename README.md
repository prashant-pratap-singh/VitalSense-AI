# VitalSense AI: Emotion & Physiological Sensing using SAM

VitalSense AI leverages the **Segment Anything Model (SAM)** combined with **MediaPipe Face Mesh** to provide advanced facial segmentation for real-time physiological monitoring and emotion sensing.

## 🚀 Features

- **Real-time Facial Segmentation**: Custom integration of SAM with MediaPipe for precise segmentation of eyes, lips, jawline, and forehead.
- **AI-Powered Analysis**: (In development) Automated calculation of Eye Aspect Ratio (EAR) and Lip Stretch for drowsiness and pain detection.
- **Premium Frontend**: Responsive, modern UI built with Next.js and Tailwind CSS.
- **FastAPI Backend**: High-performance backend handling SAM inference.

## 📂 Repository Structure

- `/backend`: FastAPI server and SAM integration logic (`sam_helper.py`).
- `/frontend`: Next.js web application with real-time webcam feed.
- `/training`: Scripts for fine-tuning SAM on custom facial datasets (`train_sam.py`, `dataset_prep.py`).

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend Setup
```powershell
cd backend
pip install -r requirements.txt
python main.py
```

### 2. Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```

### 3. Weights & Data
- Download `sam_vit_b_01ec64.pth` and place it in the project root or `training/` folder.
- The project includes `archive (1).zip` (via Git LFS) which contains the training dataset.

## 🧠 Training / Fine-tuning
To fine-tune SAM on the provided dataset:
1. Extract the dataset from `archive (1).zip`.
2. Run `python training/dataset_prep.py` to generate facial masks.
3. Run `python training/train_sam.py` to start the fine-tuning process.

## 📜 License
This project is licensed under the MIT License.
