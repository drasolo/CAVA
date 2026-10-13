import numpy as np
from sklearn import metrics
from cava_practice.testing import check, close, forbid


def test(m):
    forbid(m, ["sklearn"])
    r = np.random.default_rng(1)
    labels = ["fred", "daphne", "velma", "shaggy"]
    yt = r.choice(labels, 200)
    yp = np.where(r.random(200) < 0.7, yt, r.choice(labels, 200))
    close(m.accuracy(yt, yp), metrics.accuracy_score(yt, yp), 1e-12, "accuracy")
    cm = np.asarray(m.confusion_matrix(yt, yp, labels))
    ref = metrics.confusion_matrix(yt, yp, labels=labels)
    check(np.array_equal(cm, ref), f"confusion_matrix wrong (rows must be TRUE labels):\n{cm}\nexpected\n{ref}")
    for lab in labels:
        p, rc = m.precision_recall(yt, yp, lab)
        want = [metrics.precision_score(yt, yp, labels=[lab], average="micro"),
                metrics.recall_score(yt, yp, labels=[lab], average="micro")]
        close([p, rc], want, 1e-12, f"precision/recall for {lab}")
    check(tuple(m.precision_recall(["a", "a"], ["b", "b"], "a")) == (0.0, 0.0),
          "no positive predictions: precision must be 0.0, not a crash")
