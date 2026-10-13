"""Average precision, exactly as Tema 2 is graded

Course: Tema 2 (Task 1 graded by AP, Task 2 by mAP). Rank ALL detections from
all images by score; walk down the list; each detection is a TP if it hits a
not-yet-matched ground-truth face with IoU >= 0.3, otherwise FP (a second hit
on the same face is a duplicate = FP). Precision/recall after each detection
give the curve; AP is the area under it.

TASK
  average_precision(detections, scores, file_names, gt) -> float
      detections: (N, 4) inclusive boxes; scores: (N,); file_names: (N,) str
      gt: list of (file_name, (xmin, ymin, xmax, ymax)).
      Match exactly what evalueaza_solutie.py does:
        1. sort detections by score, descending (np.argsort(scores)[::-1]);
        2. for each, among the GT boxes of ITS image take the one with max IoU
           (inclusive +1 IoU, d70); TP if IoU >= 0.3 and that GT is unmatched
           (then mark it matched); else FP;
        3. rec = cumsum(TP) / len(gt); prec = cumsum(TP) / (cumsum(TP) + cumsum(FP));
        4. m_rec = [0, rec..., 1], m_pre = [0, prec..., 0];
           i = indices where m_rec changes (i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1);
           AP = sum((m_rec[i] - m_rec[i - 1]) * m_pre[i]).
      NOTE: the official code also has a loop meant to make precision
      monotonic, but its range() is written backwards and never runs. Your
      score is computed WITHOUT that step, so don't add it.

QUESTION once it passes: why must every detection carry a score? What does the
curve look like if all scores are equal?
"""


def average_precision(detections, scores, file_names, gt):
    raise NotImplementedError
