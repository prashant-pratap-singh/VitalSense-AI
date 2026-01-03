import cv2
import os
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import mediapipe as mp
from tqdm import tqdm

def generate_masks(image_dir, output_mask_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use MediaPipe to get initial landmarks for "pseudo-labels"
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Generating pseudo-masks for {len(image_files)} images...")
    
    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Create a simple mask for the face region or specific features
            # Here we combine lips and eyes as a target for fine-tuning
            
            # Example: Lips indices
            lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            pts = []
            for idx in lips_indices:
                pt = landmarks.landmark[idx]
                pts.append([int(pt.x * w), int(pt.y * h)])
            
            cv2.fillPoly(mask, [np.array(pts)], 255)
            
            # Save mask
            mask_path = os.path.join(output_mask_dir, filename)
            cv2.imwrite(mask_path, mask)

if __name__ == "__main__":
    # Example for Depression Data
    image_base = r"C:\Users\prash\Downloads\SAM_Dataset\Data\Depression Data\data\train"
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprize"]
    
    for emotion in emotions:
        input_dir = os.path.join(image_base, emotion)
        output_dir = os.path.join(image_base, emotion + "_Masks")
        if os.path.exists(input_dir):
            generate_masks(input_dir, output_dir)
