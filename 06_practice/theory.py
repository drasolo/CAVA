"""Review: the theory questions and your own drill cards, on a Leitner schedule.

    python theory.py            up to 10 items: what's due first, then new ones
    python theory.py 25         up to 25
    python theory.py --due      how many items are due today (the coach uses this)
    python theory.py --stats    how you're doing per topic, weakest first

Answer out loud or on paper first, THEN press Enter to see the model answer,
then say honestly whether you had it (y/n). Right moves an item up a box
(back in 3 days, 1 week, 2 weeks, 1 month); wrong sends it back to tomorrow.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cava_practice import leitner, state  # noqa: E402

# (topic, question, answer)
QUESTIONS = [
    ("filters", "What is the difference between correlation and convolution? When does it not matter?",
     "Convolution flips the kernel on both axes before sliding it; correlation doesn't. For symmetric kernels "
     "(box, Gaussian) the result is identical. cv2.filter2D computes correlation."),
    ("filters", "Why is the Gaussian separable, and why does it matter?",
     "G(x, y) = g(x) g(y), so a k x k blur = a 1D blur along rows then along columns: 2k operations per pixel "
     "instead of k^2."),
    ("filters", "How do you pick the Gaussian kernel size for a given sigma?",
     "About 6*sigma rounded up to odd (covers +-3 sigma, >99% of the mass). Too small truncates the Gaussian."),
    ("filters", "Salt-and-pepper noise: mean filter or median filter? Why?",
     "Median: the outlier pixels never become the median of a neighbourhood, so they vanish and edges stay "
     "sharp. A mean spreads each outlier into a grey smudge."),
    ("filters", "Box filter 3x3 applied to a single bright pixel of value 9 on black: what is the output?",
     "A 3x3 square of 1s (9/9) centred on the pixel. Filtering an impulse reproduces the (flipped) kernel."),
    ("gradients", "Write the Sobel x kernel. What does a large positive gx mean?",
     "[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]. Intensity increases from left to right at that pixel."),
    ("gradients", "Gradient magnitude and orientation from gx, gy?",
     "magnitude = sqrt(gx^2 + gy^2), orientation = atan2(gy, gx). The gradient points toward increasing "
     "intensity, perpendicular to the edge."),
    ("edges", "List the steps of Canny.",
     "1) Gaussian smoothing, 2) gradient magnitude and orientation, 3) non-maximum suppression along the "
     "gradient direction (thin edges), 4) hysteresis with two thresholds: strong edges kept, weak edges kept "
     "only if connected to strong ones."),
    ("edges", "Why two thresholds (hysteresis) in Canny instead of one?",
     "One high threshold breaks edges into pieces; one low threshold lets noise in. Hysteresis keeps weak "
     "pixels only when they connect to strong ones, so edges stay continuous without the noise."),
    ("hough", "Why does Hough use (rho, theta) instead of y = mx + b?",
     "m is infinite for vertical lines and unbounded in general. rho = x cos t + y sin t has bounded "
     "parameters: t in [0, 180), |rho| <= image diagonal."),
    ("hough", "In (rho, theta) space, what does a single image point vote for? And a line?",
     "A point votes along a sinusoid rho(t) = x cos t + y sin t. Points on the same image line give sinusoids "
     "that intersect at one (rho, theta): the line."),
    ("hough", "Hough for circles with unknown radius: parameter space? How does the gradient help?",
     "3D (a, b, r): each edge point votes on a cone. Knowing the gradient direction, the centre lies along "
     "the gradient line, so each point votes along a line instead of a whole circle per radius."),
    ("hough", "Two extensions of the basic Hough line algorithm from the course?",
     "1) use the gradient orientation at each edge point instead of iterating over every theta; 2) weight "
     "each vote by the gradient magnitude."),
    ("hough", "What is the generalised Hough transform?",
     "Detects arbitrary shapes: from training, store for each visual word / edge orientation the displacement "
     "to the object centre; at test time each match votes for a centre; peaks are detections."),
    ("ransac", "Describe RANSAC for fitting a line.",
     "Repeat N times: pick 2 random points, fit the line, count inliers (distance < t). Keep the line with the "
     "most inliers, optionally refit on all its inliers with least squares."),
    ("ransac", "How many RANSAC iterations for p = 0.99, inlier ratio r = 0.5, k = 2 points per sample?",
     "N = log(1 - p) / log(1 - r^k) = log(0.01) / log(0.75) = 16.01 -> 17 iterations."),
    ("ransac", "When does RANSAC work badly?",
     "Very low inlier ratio (r^k tiny means huge N) and models needing many points per sample (large k)."),
    ("binary", "What does Otsu's method optimise?",
     "It picks the threshold maximising the between-class variance w0 w1 (mu0 - mu1)^2 of the histogram "
     "(equivalently minimising the within-class variance)."),
    ("binary", "Opening vs closing: definition and use?",
     "Opening = erosion then dilation: removes small white specks, keeps big shapes. Closing = dilation then "
     "erosion: fills small black holes and gaps."),
    ("binary", "4- vs 8-connectivity: two pixels touching only at a corner?",
     "Same component with 8-connectivity, different components with 4-connectivity."),
    ("binary", "Circularity of a shape: formula and value for a circle?",
     "4 pi A / P^2. Equal to 1 for a perfect circle, smaller for anything else (square ~0.785)."),
    ("color", "Why use HSV instead of RGB for colour thresholds? What's the catch with red?",
     "Hue separates colour from brightness, so one hue range survives lighting changes. Red wraps around: in "
     "OpenCV H is 0..179 and red sits near 0 AND near 179, so it needs two ranges."),
    ("shape", "How do you compare two contours/shapes in a way invariant to scale and rotation?",
     "Hu moments (cv2.matchShapes), or chamfer matching with a distance transform. Template matching is "
     "neither rotation- nor scale-invariant."),
    ("shape", "What is chamfer matching?",
     "Compute the distance transform of the image edge map; the cost of a template placement is the average "
     "distance-transform value under the template's edge points. Low cost = edges line up."),
    ("matching", "Why normalised cross-correlation instead of plain correlation for template matching?",
     "Plain correlation is largest on bright regions regardless of shape. NCC subtracts means and divides by "
     "the norms, giving a score in [-1, 1] invariant to brightness and contrast."),
    ("geometry", "How many point correspondences define a homography? Why?",
     "4 (no 3 collinear). H is 3x3 up to scale = 8 unknowns; each correspondence gives 2 equations."),
    ("geometry", "What is the difference between an affine transform and a homography?",
     "Affine (6 params, 3 points) keeps parallel lines parallel. A homography (8 params, 4 points) models "
     "perspective: parallel lines may meet. A photographed board needs a homography."),
    ("hog", "Describe the HOG pipeline (Dalal-Triggs).",
     "(Gamma/colour normalisation), gradients, per-cell histograms of orientation weighted by magnitude, "
     "contrast normalisation over overlapping blocks of cells, concatenate, linear SVM on fixed-size windows."),
    ("hog", "Why overlapping blocks in HOG?",
     "Each cell gets normalised relative to several neighbourhoods, which makes the descriptor robust to local "
     "illumination and contrast changes. Costs feature length."),
    ("hog", "HOG of a 64x64 window, 8x8 cells, 2x2 blocks, 9 bins: descriptor length?",
     "8x8 cells -> 7x7 blocks (stride 1 cell) x 2x2 cells x 9 bins = 49 x 36 = 1764."),
    ("detection", "Sliding-window detection: how do you find faces of different sizes?",
     "Keep the window size fixed (what the classifier was trained on) and run it on an image pyramid "
     "(rescaled copies), mapping detections back by dividing coordinates by the scale."),
    ("detection", "What is hard negative mining and why do it?",
     "Run the trained detector on face-free images, collect the false positives with high scores, add them "
     "as negatives, retrain. Random negatives are too easy; this teaches the classifier the confusing cases."),
    ("detection", "What does non-maximum suppression do in detection?",
     "Sort detections by score; keep the best, drop those overlapping it by IoU above a threshold; repeat. "
     "Removes duplicate detections of the same object."),
    ("detection", "Define precision, recall, and average precision.",
     "Precision = TP / (TP + FP), recall = TP / (all ground truths). Sorting detections by score and sweeping "
     "the threshold traces the precision-recall curve; AP is the area under it."),
    ("detection", "Why must detections have a score to compute AP?",
     "AP sweeps a threshold over the ranking. Without scores there is no ranking: the curve is a single point "
     "and the area is meaningless."),
    ("detection", "IoU of two 10x10 boxes overlapping by half their width?",
     "Intersection 50, union 150, IoU = 1/3."),
    ("bow", "Bag of visual words: the steps?",
     "Extract local descriptors (SIFT) from training images; k-means them into k visual words; describe each "
     "image as the histogram of its descriptors' nearest words; train a classifier on the histograms."),
    ("bow", "What does the spatial pyramid add to bag of words?",
     "Histograms also over 2x2, 4x4 sub-regions concatenated (weighted), putting back coarse spatial layout "
     "that a single histogram throws away."),
    ("bow", "SIFT descriptor in one sentence?",
     "A 16x16 patch around a keypoint, split into 4x4 cells, each an 8-bin gradient-orientation histogram: "
     "128 numbers, rotated to the dominant orientation and normalised."),
    ("cnn", "Output size of a conv layer: input 32x32, kernel 5x5, stride 1, padding 0? With padding 2?",
     "(32 - 5 + 2p) / s + 1: 28x28 without padding, 32x32 with padding 2."),
    ("cnn", "Parameters of a conv layer: 3 input channels, 16 filters of 5x5, with bias?",
     "16 x (5 x 5 x 3) + 16 = 1216."),
    ("cnn", "Why do ResNet's skip connections help?",
     "Each block learns a residual F(x) added to x, so gradients flow through the identity path: very deep "
     "networks train without degrading."),
    ("cnn", "How can you use a pretrained CNN on a small dataset (e.g. 160 exam images)?",
     "Use it as a fixed feature extractor (take the penultimate-layer activations, train a linear "
     "classifier/SVM on them), or fine-tune only the last layers."),
    ("autoencoder", "What is an autoencoder and how is it trained?",
     "Encoder compresses the input to a low-dimensional code, decoder reconstructs it; trained to minimise "
     "reconstruction error. Unsupervised. Denoising version: corrupt the input, reconstruct the clean one."),
    ("autoencoder", "PCA vs autoencoder?",
     "A linear autoencoder with MSE loss learns the same subspace as PCA. Nonlinear autoencoders can capture "
     "curved manifolds PCA can't."),
    ("seam", "Seam carving: what's a seam and how is the optimal one found?",
     "A connected path of one pixel per row (vertical seam). Energy = gradient magnitude; dynamic programming "
     "M[i, j] = e[i, j] + min(M[i-1, j-1..j+1]); backtrack from the minimum in the last row; remove it."),
    ("video", "Background subtraction: how do you model the background, and how is foreground decided?",
     "Background B = mean or median of frames (median resists moving objects), or a running average. "
     "Foreground where |I_t - B| > threshold, cleaned with morphology."),
    ("ml", "Why scale features before an SVM, and where do you fit the scaler?",
     "SVMs depend on distances/dot products, so large-range features dominate. Fit the scaler on training data "
     "only (inside each CV fold), then apply it to validation/test."),
    ("ml", "Linear SVM: what does C control?",
     "The trade-off between a wide margin and training errors. Small C: wider margin, more regularisation. "
     "Large C: fits training data harder, risk of overfitting."),
]


def items():
    """Every reviewable item as (key, topic, question, answer): the bank plus your cards."""
    out = [(q, t, q, a) for t, q, a in QUESTIONS]
    for stem, c in sorted(state.load(state.CARDS).items()):
        out.append((f"card:{stem}", c.get("topic", f"card {stem[:3]}"), c["q"], c["a"]))
    return out


def main():
    stats = leitner.migrate(state.load(state.THEORY))
    today = state.today()
    all_items = items()
    if "--due" in sys.argv:
        print(len(leitner.due_keys(stats, [k for k, *_ in all_items], today)))
        return
    if "--stats" in sys.argv:
        topics = {}
        for key, t, _, _ in all_items:
            s = stats.get(key, {})
            agg = topics.setdefault(t, [0, 0, 0])
            agg[0] += 1
            agg[1] += s.get("seen", 0)
            agg[2] += s.get("right", 0)
        for t, (n, seen, right) in sorted(topics.items(), key=lambda kv: (kv[1][2] / kv[1][1]) if kv[1][1] else -1):
            rate = f"{right / seen:.0%}" if seen else "not yet"
            print(f"  {t:12s} {n:2d} items   answered {seen:3d}x   right {rate}")
        return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keys = leitner.pick(stats, [k for k, *_ in all_items], today, n, random.Random())
    by_key = {k: (t, q, a) for k, t, q, a in all_items}
    if not keys:
        print("Nothing due and nothing new. Come back tomorrow, or run a drill.")
        return
    score = 0
    for k, key in enumerate(keys, 1):
        topic, q, a = by_key[key]
        print()
        print(f"[{k}/{len(keys)}] ({topic}) {q}")
        input("   ... answer it yourself, then press Enter ")
        print(f"   {a}")
        ok = input("   Did you have it? [y/n] ").strip().lower().startswith("y")
        leitner.answer(stats, key, ok, today)
        score += ok
        state.save(state.THEORY, stats)
        state.touch()
    left = len(leitner.due_keys(stats, list(by_key), today))
    print()
    print(f"{score}/{len(keys)}." + (f" {left} still due today." if left else " Nothing else due today."))


if __name__ == "__main__":
    main()
