import os
import cv2
from nuscenes.nuscenes import NuScenes
from config import NUSCENES_ROOT, CAMERA_CHANNEL


class NuScenesLoader:
    def __init__(self, dataroot=None):
        if dataroot is None:
            dataroot = NUSCENES_ROOT
        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=dataroot,
            verbose=True
        )

    def frames(self, camera_channel=None):
        """
        Generator: yields (frame, timestamp, token) tuples in scene order.
        
        Args:
            camera_channel: Camera channel to use (e.g., 'CAM_FRONT')
                          If None, uses CAMERA_CHANNEL from config
        """
        if camera_channel is None:
            camera_channel = CAMERA_CHANNEL
            
        for scene in self.nusc.scene:
            sample = self.nusc.get("sample", scene["first_sample_token"])

            while sample:
                cam_data = self.nusc.get("sample_data", sample["data"][camera_channel])
                img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
                img = cv2.imread(img_path)
                
                # Yield frame, timestamp, and token
                yield img, cam_data["timestamp"], sample["token"]

                sample = (
                    self.nusc.get("sample", sample["next"])
                    if sample["next"]
                    else None
                )
