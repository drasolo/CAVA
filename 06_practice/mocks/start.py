"""Start a timed mock exam.

    python mocks/start.py                 list the available exams
    python mocks/start.py 2025-06-06      create a fresh attempt and start the clock

The attempt folder mirrors the exam room: Enunt.pdf, data/ex1, data/ex2/antrenare
(with annotations), data/ex2/testare (images only), and two notebooks to fill.
Work for 2 hours, then run mocks/grade.py on the folder.
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exams import EXAMS_SPEC, ex2_annotations, ex2_split  # noqa: E402
from notebooks import write_notebooks  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from cava_practice.faces import ex1_dir  # noqa: E402


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in EXAMS_SPEC:
        print("Available mock exams:")
        for k, v in EXAMS_SPEC.items():
            print(f"  {k}  {v['title']}")
        print("\nusage: python mocks/start.py <exam>")
        return
    exam = sys.argv[1]
    spec = EXAMS_SPEC[exam]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    att = HERE / "attempts" / f"{exam}_{stamp}"
    data = att / "data"
    (data / "ex2" / "antrenare").mkdir(parents=True)
    (data / "ex2" / "testare").mkdir()
    (att / "outputs" / "ex2").mkdir(parents=True)
    sealed_dir = att / ".sealed"
    sealed_dir.mkdir()

    shutil.copy2(spec["statement"], att / "Enunt.pdf")
    shutil.copytree(ex1_dir(exam), data / "ex1")
    dev, sealed = ex2_split(exam)
    for img in dev:
        shutil.copy2(img, data / "ex2" / "antrenare" / img.name)
        for a in ex2_annotations(exam, img):
            shutil.copy2(a, data / "ex2" / "antrenare" / a.name)
    for img in sealed:
        shutil.copy2(img, data / "ex2" / "testare" / img.name)
        for a in ex2_annotations(exam, img):
            shutil.copy2(a, sealed_dir / a.name)

    write_notebooks(att, exam)
    started = dt.datetime.now().isoformat(timespec="seconds")
    (att / "attempt.json").write_text(json.dumps({"exam": exam, "started": started}, indent=2))

    print(f"Mock exam {exam}: {spec['title']}")
    print(f"Folder:   {att}")
    print(f"Started:  {started}. You have 2 hours.\n")
    print("1. Read Enunt.pdf. Ex2 here has", len(dev), "training images (the rest of the real",
          "training set is sealed as your test set:", len(sealed), "images in data/ex2/testare).")
    print("2. Fill ex1.ipynb and ex2.ipynb (jupyter notebook / VS Code).")
    print("3. Don't open the .sealed folder; it holds the test answers.")
    print(f"4. When done: python mocks/grade.py \"{att}\"")


if __name__ == "__main__":
    main()
