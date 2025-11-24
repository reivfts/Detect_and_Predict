import os
import cv2
import glob
from nuscenes.nuscenes import NuScenes
from config import NUSCENES_ROOT, CAMERA_CHANNEL


class SimpleImageLoader:
    """
    Simple loader that reads images directly from a directory.
    Use this when you have raw images without NuScenes metadata.
    """
    def __init__(self, image_directory):
        self.image_directory = image_directory
        # Get all image files sorted by name
        self.image_files = sorted(glob.glob(os.path.join(image_directory, "*.jpg"))) + \
                          sorted(glob.glob(os.path.join(image_directory, "*.png")))
        print(f"Found {len(self.image_files)} images in {image_directory}")
    
    def frames(self, camera_channel=None):
        """
        Generator: yields (frame, timestamp, token) tuples.
        For simple loader, timestamp and token are just frame indices.
        """
        for idx, img_path in enumerate(self.image_files):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image at {img_path}")
                continue
            
            # Use index as timestamp and token since we don't have metadata
            yield img, idx, str(idx)


class NuScenesLoader:
    def __init__(self, dataroot=None, version="v1.0-mini"):
        if dataroot is None:
            dataroot = NUSCENES_ROOT
        
        # If using trainval blobs without metadata, fallback to mini metadata
        # but use trainval images when available
        self.trainval_dataroot = None
        if "trainval" in dataroot and not os.path.exists(os.path.join(dataroot, version)):
            # Using trainval images but no trainval metadata
            # Fall back to mini metadata but remember trainval path for images
            self.trainval_dataroot = dataroot
            dataroot = r"C:\Users\rayra\OneDrive\Desktop\v1.0-mini"
            version = "v1.0-mini"
            print(f"Note: Using v1.0-mini metadata with trainval images from {self.trainval_dataroot}")
        
        self.nusc = NuScenes(
            version=version,
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
                
                # If using trainval images, try trainval path first
                if self.trainval_dataroot:
                    img_path = os.path.join(self.trainval_dataroot, cam_data["filename"])
                    if not os.path.exists(img_path):
                        # Fallback to original path
                        img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
                else:
                    img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
                
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not load image at {img_path}")
                    sample = (
                        self.nusc.get("sample", sample["next"])
                        if sample["next"]
                        else None
                    )
                    continue
                
                # Yield frame, timestamp, and token
                yield img, cam_data["timestamp"], sample["token"]

                sample = (
                    self.nusc.get("sample", sample["next"])
                    if sample["next"]
                    else None
                )
