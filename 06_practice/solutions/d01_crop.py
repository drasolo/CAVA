def parse_annotations(lines):
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        box = tuple(int(float(v)) for v in parts[1:5])
        out.append({"file": parts[0], "box": box, "label": parts[5] if len(parts) > 5 else None})
    return out


def crop(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    return img[y0:y1 + 1, x0:x1 + 1]
