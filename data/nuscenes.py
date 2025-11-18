import os
import cv2
from nuscenes.nuscenes import NuScenes
from config import NUScenes_PATH, CAMERA_CHANNEL


class NuScenesLoader:
    def __init__(self):
        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=NUScenes_PATH,
            verbose=True
        )

    def frames(self):
        """Generator: yields frames in correct scene order."""
        for scene in self.nusc.scene:
            sample = self.nusc.get("sample", scene["first_sample_token"])

            while sample:
                cam_data = self.nusc.get("sample_data", sample["data"][CAMERA_CHANNEL])
                img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
                img = cv2.imread(img_path)

                yield img

                sample = (
                    self.nusc.get("sample", sample["next"])
                    if sample["next"]
                    else None
                )
