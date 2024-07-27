import numpy as np
# pip install insightface==0.7.3
from insightface.app import FaceAnalysis
import os
import folder_paths

### 
# https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/165#issue-2055829543
###
class FaceAnalysis2(FaceAnalysis):
    # def __init__(self, provider="CPU", name="buffalo_l"):
    #     self.face_analysis = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
    #     self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
    #     self.thresholds = THRESHOLDS["ArcFace"]

    # NOTE: allows setting det_size for each detection call.
    # the model allows it but the wrapping code from insightface
    # doesn't show it, and people end up loading duplicate models
    # for different sizes where there is absolutely no need to
    def get(self, img, max_num=0, det_size=(640, 640)):
        if det_size is not None:
            self.det_model.input_size = det_size

        return super().get(img, max_num)

def analyze_faces(face_analysis: FaceAnalysis, img_data: np.ndarray, det_size=(640, 640)):
    # NOTE: try detect faces, if no faces detected, lower det_size until it does
    detection_sizes = [None] + [(size, size) for size in range(640, 256, -64)] + [(256, 256)]

    for size in detection_sizes:
        faces = face_analysis.get(img_data, det_size=size)
        if len(faces) > 0:
            return faces

    return []


def insightface_loader(provider):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)

    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name="buffalo_l", root=path, providers=[provider + 'ExecutionProvider',])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model
