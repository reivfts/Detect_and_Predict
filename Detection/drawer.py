import cv2
from matplotlib.pyplot import box
import numpy as np
from ultralytics.utils.plotting import colors


class Drawer:
    def __init__(self, line_width=2):
        self.line_width = line_width

    # DRAW BOUNDING BOX
    def draw_box(self, frame, box, cls_name, track_id=None):
        """
        Draws a bounding box and class label, using the track_id for consistent coloring.
        """

        x1, y1, x2, y2 = map(int, box)

        # Use track_id for color consistency if provided
        color = colors(track_id if track_id is not None else cls_name, bgr=True)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_width)

        label = f"{cls_name}"  # To include ID: f"{cls_name} ID:{track_id}"
        cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        )


    # DRAW SEGMENTATION MASK (OPTIONAL)
    def draw_mask(self, frame, mask, track_id):
        """
        Overlay a semi-transparent mask on the frame.
        """

        color = np.array(colors(track_id, bgr=True), dtype=np.uint8)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask > 0] = color

        alpha = 0.4
        cv2.addWeighted(colored_mask, alpha, frame, 1 - alpha, 0, frame)

    # DRAW ONLY ID LABEL
    def draw_id_label(self, frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)

        cv2.putText(
            frame,
            f"ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
