"""Non-maximum suppression for detections

Course: Lab 9-10, Tema 2. A sliding-window detector fires on many
overlapping windows around each face. NMS keeps the best-scoring one and
drops everything that overlaps it too much. Without it, every duplicate is a
false positive and your AP collapses.

TASK
  nms(boxes, scores, iou_thr) -> list of kept indices, highest score first.
      Greedy: take the best remaining box, keep it, drop every remaining box
      whose IoU with it is > iou_thr, repeat. Use the inclusive IoU (d70).
  nms_per_image(boxes, scores, image_ids, iou_thr) -> kept indices
      Same, but boxes only suppress boxes from the SAME image. Return the
      kept indices sorted ascending.
"""


def nms(boxes, scores, iou_thr):
    raise NotImplementedError


def nms_per_image(boxes, scores, image_ids, iou_thr):
    raise NotImplementedError
