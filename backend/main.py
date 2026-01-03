from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io
from PIL import Image
from sam_helper import VitalSAM

app = FastAPI(title="VitalSense AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
vSAM = None

@app.on_event("startup")
async def startup_event():
    global vSAM
    try:
        vSAM = VitalSAM()
    except Exception as e:
        print(f"Error loading SAM: {e}")

@app.get("/")
def read_root():
    return {"status": "VitalSense AI Backend Running"}

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    global vSAM
    if vSAM is None:
        return {"error": "Model not loaded"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    masks = vSAM.segment_face(image)
    if not masks:
        return {"error": "No face detected"}

    # Process masks into visualization
    results = {}
    overlay = image.copy()
    
    # Colors for different regions
    colors = {
        "left_eye": [255, 0, 0],
        "right_eye": [0, 255, 0],
        "lips": [0, 0, 255],
        "forehead": [255, 255, 0],
        "jawline": [255, 0, 255]
    }

    for region, mask in masks.items():
        color = colors.get(region, [255, 255, 255])
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        # Also send individual mask info if needed
        # For now, just send the combined overlay
        
    _, buffer = cv2.imencode('.png', overlay)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "image": f"data:image/png;base64,{img_str}",
        "regions": list(masks.keys()),
        "analysis": {
            "eye_aspect_ratio": 0.3, # Placeholder for drowsiness detection
            "lip_stretch": 0.5,      # Placeholder for pain estimation
            "prediction": "Normal"    # Placeholder for triage
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
