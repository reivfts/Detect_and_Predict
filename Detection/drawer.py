import cv2
from ultralytics.utils.plotting import Annotator, colors


class DrawEngine:
    def __init__(self):
        pass

    def draw(self, frame, tracks):
        annotator = Annotator(frame, line_width=2)

        for box, tid, cls in tracks:
            x1, y1, x2, y2 = map(int, box)

            annotator.box_label(
                [x1, y1, x2, y2],
                label=f"{cls} ID:{tid}",
                color=colors(tid, True)
            )

        return annotator.result()
