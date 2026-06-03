import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Geometry
# ----------------------------
def intersection_over_union(bbox_a, bbox_b) -> float:
    # bboxes are [x1, y1, x2, y2] with inclusive coords
    x_a = max(int(bbox_a[0]), int(bbox_b[0]))
    y_a = max(int(bbox_a[1]), int(bbox_b[1]))
    x_b = min(int(bbox_a[2]), int(bbox_b[2]))
    y_b = min(int(bbox_a[3]), int(bbox_b[3]))

    inter_w = max(0, x_b - x_a + 1)
    inter_h = max(0, y_b - y_a + 1)
    inter_area = inter_w * inter_h

    area_a = max(0, int(bbox_a[2]) - int(bbox_a[0]) + 1) * max(0, int(bbox_a[3]) - int(bbox_a[1]) + 1)
    area_b = max(0, int(bbox_b[2]) - int(bbox_b[0]) + 1) * max(0, int(bbox_b[3]) - int(bbox_b[1]) + 1)

    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / float(denom)


def compute_average_precision(rec, prec) -> float:
    # Pascal VOC style AP
    m_rec = np.concatenate(([0.0], rec, [1.0]))
    m_pre = np.concatenate(([0.0], prec, [0.0]))

    # make precision monotonically decreasing
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])

    # integrate
    idx = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    ap = np.sum((m_rec[idx] - m_rec[idx - 1]) * m_pre[idx])
    return float(ap)


# ----------------------------
# Task 1 evaluation (all faces)
# ----------------------------
def eval_detections(detections, scores, file_names, ground_truth_path, iou_thr: float = 0.3) -> float:
    """
    Evaluate detections against GT and save PR curve image.
    Prints a compact summary and returns AP.

    Inputs:
      detections: (N,4) [x1,y1,x2,y2]
      scores:     (N,)
      file_names: (N,) strings, must match GT filenames exactly (e.g. '0001.jpg')
      ground_truth_path: txt with columns: filename x1 y1 x2 y2 [optional extras...]
    """
    gt = np.loadtxt(ground_truth_path, dtype=str)
    if gt.ndim == 1:
        gt = gt[None, :]  # single line file

    gt_file_names = gt[:, 0].astype(str)
    gt_boxes = gt[:, 1:5].astype(int)  # only first 4 coords; ignore extra columns (character)

    num_gt = len(gt_boxes)
    if num_gt == 0:
        print("[Task1] No GT boxes found. Check ground_truth_path.")
        return 0.0

    # mark which GT boxes already matched
    gt_matched = np.zeros(num_gt, dtype=np.uint8)

    detections = np.asarray(detections, dtype=int)
    scores = np.asarray(scores, dtype=float)
    file_names = np.asarray(file_names, dtype=str)

    # sort by descending score
    order = np.argsort(scores)[::-1]
    detections = detections[order]
    scores = scores[order]
    file_names = file_names[order]

    n = len(detections)
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)
    dup = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        fn = file_names[i]
        bbox = detections[i]

        gt_idx_on_img = np.where(gt_file_names == fn)[0]
        if gt_idx_on_img.size == 0:
            fp[i] = 1.0
            continue

        # find best IoU for this detection among GT on the same image
        max_iou = -1.0
        best_global_idx = -1
        for j, global_gt_idx in enumerate(gt_idx_on_img):
            iou = intersection_over_union(bbox, gt_boxes[global_gt_idx])
            if iou > max_iou:
                max_iou = iou
                best_global_idx = int(global_gt_idx)

        if max_iou >= iou_thr and best_global_idx >= 0:
            if gt_matched[best_global_idx] == 0:
                tp[i] = 1.0
                gt_matched[best_global_idx] = 1
            else:
                fp[i] = 1.0
                dup[i] = 1
        else:
            fp[i] = 1.0

    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)

    rec = cum_tp / float(num_gt)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = compute_average_precision(rec, prec)

    print(f"[Task1] detections={n}  GT={num_gt}  matched={int(gt_matched.sum())}  duplicates={int(dup.sum())}")
    print(f"[Task1] Average Precision (AP) = {ap:.3f}")

    plt.figure()
    plt.plot(rec, prec, "-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"All faces: average precision {ap:.3f}")
    plt.grid(True, alpha=0.3)
    plt.savefig("precizie_medie_all_faces.png", dpi=150)
    plt.close()

    return ap


# ----------------------------
# Task 2 evaluation (per character)
# ----------------------------
def eval_detections_character(detections, scores, file_names, ground_truth_path, character, iou_thr: float = 0.3) -> float:
    gt = np.loadtxt(ground_truth_path, dtype=str)
    if gt.ndim == 1:
        gt = gt[None, :]

    gt_file_names = gt[:, 0].astype(str)
    gt_boxes = gt[:, 1:5].astype(int)

    num_gt = len(gt_boxes)
    if num_gt == 0:
        print(f"[Task2:{character}] No GT boxes found.")
        return 0.0

    gt_matched = np.zeros(num_gt, dtype=np.uint8)

    detections = np.asarray(detections, dtype=int)
    scores = np.asarray(scores, dtype=float)
    file_names = np.asarray(file_names, dtype=str)

    order = np.argsort(scores)[::-1]
    detections = detections[order]
    scores = scores[order]
    file_names = file_names[order]

    n = len(detections)
    tp = np.zeros(n, dtype=np.float32)
    fp = np.zeros(n, dtype=np.float32)
    dup = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        fn = file_names[i]
        bbox = detections[i]

        gt_idx_on_img = np.where(gt_file_names == fn)[0]
        if gt_idx_on_img.size == 0:
            fp[i] = 1.0
            continue

        max_iou = -1.0
        best_global_idx = -1
        for global_gt_idx in gt_idx_on_img:
            iou = intersection_over_union(bbox, gt_boxes[global_gt_idx])
            if iou > max_iou:
                max_iou = iou
                best_global_idx = int(global_gt_idx)

        if max_iou >= iou_thr and best_global_idx >= 0:
            if gt_matched[best_global_idx] == 0:
                tp[i] = 1.0
                gt_matched[best_global_idx] = 1
            else:
                fp[i] = 1.0
                dup[i] = 1
        else:
            fp[i] = 1.0

    cum_fp = np.cumsum(fp)
    cum_tp = np.cumsum(tp)

    rec = cum_tp / float(num_gt)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = compute_average_precision(rec, prec)

    print(f"[Task2:{character}] detections={n}  GT={num_gt}  matched={int(gt_matched.sum())}  duplicates={int(dup.sum())}")
    print(f"[Task2:{character}] Average Precision (AP) = {ap:.3f}")

    plt.figure()
    plt.plot(rec, prec, "-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{character} faces: average precision {ap:.3f}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"precizie_medie_{character}.png", dpi=150)
    plt.close()

    return ap


# ----------------------------
# Load helpers (match original evaluator interface)
# ----------------------------
def evaluate_results_task1(solution_path, ground_truth_path, verbose=True):
    detections = np.load(solution_path + "detections_all_faces.npy", allow_pickle=True)
    scores = np.load(solution_path + "scores_all_faces.npy", allow_pickle=True)
    file_names = np.load(solution_path + "file_names_all_faces.npy", allow_pickle=True).astype(str)

    if verbose:
        print(detections.shape)
        print(scores.shape)
        print(file_names.shape)

    return eval_detections(detections, scores, file_names, ground_truth_path)


def evaluate_results_task2(solution_path, ground_truth_path, character, verbose=True):
    detections = np.load(solution_path + f"detections_{character}.npy", allow_pickle=True)
    scores = np.load(solution_path + f"scores_{character}.npy", allow_pickle=True)
    file_names = np.load(solution_path + f"file_names_{character}.npy", allow_pickle=True).astype(str)

    if verbose:
        print(detections.shape)
        print(scores.shape)
        print(file_names.shape)

    return eval_detections_character(detections, scores, file_names, ground_truth_path, character)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    verbose = True

    solution_path_root = "../fisiere_solutie/341_Solomon_Dragos/"
    ground_truth_path_root = "../../validare/"

    # Task 1
    solution_path = solution_path_root + "task1/"
    ground_truth_path = ground_truth_path_root + "task1_gt_validare.txt"
    evaluate_results_task1(solution_path, ground_truth_path, verbose)

    # Task 2 (disabled to avoid crashing if you didn't generate Task2 files yet)
    # solution_path = solution_path_root + "task2/"
    # ground_truth_path = ground_truth_path_root + "task2_daphne_gt_validare.txt"
    # evaluate_results_task2(solution_path, ground_truth_path, "daphne", verbose)
    # ground_truth_path = ground_truth_path_root + "task2_fred_gt_validare.txt"
    # evaluate_results_task2(solution_path, ground_truth_path, "fred", verbose)
    # ground_truth_path = ground_truth_path_root + "task2_shaggy_gt_validare.txt"
    # evaluate_results_task2(solution_path, ground_truth_path, "shaggy", verbose)
    # ground_truth_path = ground_truth_path_root + "task2_velma_gt_validare.txt"
    # evaluate_results_task2(solution_path, ground_truth_path, "velma", verbose)
