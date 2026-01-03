import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import mediapipe as mp
import os
import requests

class VitalSAM:
    def __init__(self, model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        
        # Ensure model exists
        if not os.path.exists(checkpoint_path):
            print(f"Downloading {model_type} model...")
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_path}"
            response = requests.get(url, stream=True)
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Init MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_facial_prompts(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        
        # Define region mapping (MediaPipe indices)
        # Simplified for demonstration
        regions = {
            "left_eye": [33, 160, 158, 133, 153, 144],
            "right_eye": [362, 385, 387, 263, 373, 380],
            "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            "forehead": [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109],
            "jawline": [172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]
        }
        
        prompts = {}
        for name, indices in regions.items():
            pts = []
            for idx in indices:
                landmark = landmarks.landmark[idx]
                pts.append([int(landmark.x * w), int(landmark.y * h)])
            prompts[name] = np.array(pts)
            
        return prompts

    def segment_face(self, image):
        self.predictor.set_image(image)
        prompts = self.get_facial_prompts(image)
        
        if not prompts:
            return None
        
        results = {}
        for region_name, points in prompts.items():
            # Use points as prompts
            # SAM expects points as (N, 2) and labels as (N,)
            input_point = points
            input_label = np.ones(len(points))
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            # Take the best mask
            results[region_name] = masks[np.argmax(scores)]
            
        return results
