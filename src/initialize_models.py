from asd4hri.asd import Architecture_Options, ASD
from asd4hri.processors import SHAPE_PREDICTOR_68_FACE_LANDMARKS, FaceDetectorRetinaFace, FaceDetectorYN
# initialize all the models for DetectVVAD once so they will be automatically downloaded. 
for model in Architecture_Options:
    ASD(architecture=model)
    print(f"Initialized {model} for ASD...")

SHAPE_PREDICTOR_68_FACE_LANDMARKS()
print(f"Initialized face landmarks model...")


for model in [FaceDetectorRetinaFace, FaceDetectorYN]:
    init_model = model()
    print(f"Initialized {init_model.name} for face detection...")
