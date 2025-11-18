import os
from nuscenes.nuscenes import NuScenes
from config import NUSCENES_ROOT


class NuScenesLoader:
    def __init__(self):
        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=NUSCENES_ROOT,
            verbose=True
        )

    def load_scene_frames(self, camera_channel):
        for scene in self.nusc.scene:
            token = scene["first_sample_token"]
            sample = self.nusc.get("sample", token)

            while sample:
                cam_data = self.nusc.get("sample_data", sample["data"][camera_channel])
                img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
                yield img_path

                if sample["next"]:
                    sample = self.nusc.get("sample", sample["next"])
                else:
                    break
