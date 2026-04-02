"""
Computer Vision & OpenCV — Seeing the World Through Code
=========================================================

Computer vision is the field of artificial intelligence that enables
computers to interpret and understand visual information from the world —
images, video, and multi-dimensional data. It is one of the oldest and
most mathematically rich branches of AI, drawing from signal processing,
linear algebra, statistics, differential geometry, and deep learning.

OpenCV (Open Source Computer Vision Library) was initiated by Gary Bradski
at Intel in 1999 and open-sourced in 2000. It is now the world's largest
computer vision library with over 2,500 optimised algorithms, 47,000+ GitHub
stars, and over 18 million downloads per month. OpenCV is the foundation
of virtually every real-time computer vision system in the world — from
autonomous vehicles to medical imaging to industrial quality control.

The Python bindings (cv2) make OpenCV exceptionally accessible: the full
performance of C++ SIMD-optimised kernels with the expressiveness of Python.
Most operations run in microseconds on CPU and in nanoseconds on GPU via
CUDA backends. This combination of performance and accessibility makes
OpenCV the undisputed standard for computer vision engineering.

Understanding computer vision requires understanding two complementary
paradigms:
    CLASSICAL CV:  hand-engineered algorithms based on mathematical insight.
                   Filters, edges, corners, contours, transformations.
                   Fast, interpretable, no training data needed.
    DEEP CV:       learned representations via convolutional neural networks.
                   Feature extraction, classification, detection, segmentation.
                   More powerful but needs data and compute.

Both paradigms are essential. Classical CV handles the preprocessing,
calibration, geometric transformations, and real-time processing that
deep learning depends on. Deep learning handles the semantic understanding
that classical CV cannot achieve. Production systems use both.

This module covers the complete computer vision stack: image fundamentals
and the colour model hierarchy, spatial filtering and convolution theory,
edge detection from first principles (Sobel, Canny), morphological operations,
feature detection (Harris corners, SIFT, ORB), optical flow and motion
analysis, contour analysis and shape description, geometric transforms and
homography, and the classical-to-deep learning bridge via HOG descriptors
and object detection pipelines.

"""

import textwrap
import re

TOPIC_NAME   = "Computer Vision & OpenCV — Seeing the World Through Code"
DISPLAY_NAME = "11 · Computer Vision & OpenCV"
ICON         = "👁️"
SUBTITLE     = "From Pixel Mathematics to Feature Detection and Motion Analysis"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — IMAGES AS DATA: PIXELS, CHANNELS, AND COLOUR SPACES

### What Is a Digital Image?

    A digital image is a 2D or 3D array of numbers. Each number represents
    the light intensity at one spatial location (a pixel).

    Grayscale image:     H × W array, dtype=uint8, values in [0, 255]
                         0 = black, 255 = white, middle = grey
    Colour image (BGR):  H × W × 3 array, dtype=uint8
                         Three channels: Blue, Green, Red
    Float image:         H × W × C array, dtype=float32, values in [0.0, 1.0]
                         Required for many numerical operations

    CRITICAL: OpenCV uses BGR channel order, NOT RGB.
    PIL/matplotlib use RGB. When mixing these libraries, ALWAYS convert:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

### Why Images Are NumPy Arrays

    In Python, OpenCV images are NumPy ndarray objects. This means:
        - Every NumPy operation works on images
        - Slicing: roi = image[y1:y2, x1:x2] — region of interest
        - Masking: image[mask > 0] = [255, 0, 0] — selective painting
        - Arithmetic: blend = (0.5 * img1 + 0.5 * img2).astype(np.uint8)
        - Broadcasting: image + np.array([10, 0, -10]) — channel-wise shift

    Coordinate system (important convention):
        image[y, x]     — rows first (y), then columns (x)
        image.shape     — (height, width, channels)
        cv2.rectangle takes (x1, y1, x2, y2) — OpenCV uses (x, y) order
        numpy slice is [y1:y2, x1:x2]         — numpy uses (row, col) order

    This row/col vs x/y inconsistency is the source of most OpenCV bugs.
    Memorise it: NumPy = [row, col] = [y, x]; OpenCV functions = (x, y).

### Colour Spaces: Different Ways to Represent Colour

    BGR (Blue-Green-Red):
        The native OpenCV format. Direct hardware representation.
        Each channel: 0–255. Perceptually non-uniform (equal RGB steps
        ≠ equal perceived colour changes).

    HSV (Hue-Saturation-Value):
        H: colour type (0–179 in OpenCV, represents 0–360°)
           0°=red, 60°=yellow, 120°=green, 180°=cyan, 240°=blue, 300°=magenta
        S: colour purity (0=grey, 255=full colour)
        V: brightness (0=black, 255=bright)

        WHY HSV IS ESSENTIAL FOR COLOUR DETECTION:
            "Find all red objects in the image" is HARD in BGR:
            red pixels have high R and low G/B, but exact values vary
            enormously with lighting.
            In HSV, red is H≈0 or H≈160, regardless of lighting.

            bgr_mask = ((img[:,:,2] > 150) & (img[:,:,0] < 50))  # fragile
            hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0,   120, 70])
            upper = np.array([10,  255, 255])
            mask  = cv2.inRange(hsv, lower, upper)               # robust

    LAB (Lightness-A-B):
        L: perceptual lightness (0=black, 100=white)
        A: green–red axis (-128 to 127)
        B: blue–yellow axis (-128 to 127)

        WHY LAB IS ESSENTIAL FOR PERCEPTUAL TASKS:
            LAB is designed to be perceptually uniform — equal numerical
            distances correspond to equal perceived colour differences.
            The Euclidean distance in LAB space approximates human colour
            perception:  ΔE = sqrt(ΔL² + Δa² + Δb²)

            Use LAB for:
            - Skin detection (consistent across skin tones and lighting)
            - Background subtraction
            - Colour constancy correction
            - Image colour matching and transfer

    YCrCb (Luma-Chroma Red-Chroma Blue):
        Y:  luma (brightness, the greyscale component)
        Cr: chrominance red (colour difference)
        Cb: chrominance blue (colour difference)

        Used in JPEG and video compression — separates brightness from colour
        because the human eye is far more sensitive to luminance than chrominance.
        Excellent for face detection and skin colour segmentation:
        skin pixels cluster tightly in CrCb space.

    Greyscale:
        Weighted average of BGR channels:
        Y = 0.114×B + 0.587×G + 0.299×R
        The green channel gets highest weight (most perceptual luminance).

    Choosing the right colour space:
        Detecting coloured objects:     HSV
        Perceptual colour distance:     LAB
        Brightness/colour separation:   YCrCb
        Skin detection:                 YCrCb or HSV
        General processing:             Grayscale
        Hardware output:                BGR

### Histograms: The Statistical Fingerprint of an Image

    A histogram counts pixel frequency at each intensity value (0–255).
    It reveals: exposure, contrast, dynamic range, colour distribution.

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # [image]: list of images
    # [0]:     channel index (0=B, 1=G, 2=R for BGR; 0=Grey for greyscale)
    # None:    no mask (full image)
    # [256]:   number of bins
    # [0,256]: pixel value range

    Histogram equalisation:
        Redistributes pixel intensities to span the full [0, 255] range.
        Dramatically improves contrast in dark or overexposed images.
        equal = cv2.equalizeHist(grey)

    CLAHE (Contrast Limited Adaptive Histogram Equalisation):
        Equalises in small tiles independently, then blends.
        Avoids over-amplifying noise in already-good regions.
        Superior to global equalisation for natural images.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        result = clahe.apply(grey)


##### PART 2 — SPATIAL FILTERING AND CONVOLUTION

### What Is Convolution?

    Convolution is the fundamental operation of image processing. A kernel
    (small matrix of weights) slides over the image. At each position, it
    computes a weighted sum of the pixels it overlaps. The result is a new
    image where each pixel reflects a neighbourhood relationship.

    Mathematical definition (2D discrete convolution):
        (I * K)[x, y] = Σᵢ Σⱼ  I[x+i, y+j] × K[i, j]

    The kernel encodes what property to measure:
        Averaging kernel  → smooth/blur (low-frequency enhancement)
        Sharpening kernel → amplify differences (high-frequency enhancement)
        Gradient kernel   → measure intensity change = edges
        Identity kernel   → no change

    OpenCV:
        result = cv2.filter2D(image, -1, kernel)
        # -1: output has same depth as input

### Blur Filters: Noise Reduction

    Box filter (simple averaging):
        kernel = np.ones((k, k), np.float32) / (k * k)
        # Every pixel in the neighbourhood equally weighted
        # Fast but blurs edges significantly

    Gaussian blur (most common):
        kernel = Gaussian function of distance from centre
        G(x, y) = (1/2πσ²) × exp(-(x²+y²)/2σ²)
        # Closer pixels have more influence than distant ones
        # The most natural blur — matches how optics focus
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX)

    Parameters:
        ksize:  kernel size (must be odd: 3, 5, 7, ...). Larger = more blur.
        sigmaX: standard deviation. Rule: sigma ≈ 0.3×((ksize-1)/2 - 1) + 0.8
                If 0: computed automatically from ksize.

    Median blur (best for salt-and-pepper noise):
        Replaces each pixel with the MEDIAN of its neighbourhood.
        The median is naturally outlier-resistant — no single pixel
        dominates, so isolated noise pixels get eliminated.
        blurred = cv2.medianBlur(image, ksize)   # ksize must be odd

        WHY MEDIAN OUTPERFORMS GAUSSIAN ON IMPULSE NOISE:
        One bright pixel in a neighbourhood of 9:
        Gaussian: (255 + 8×50) / 9 = 73  (noise propagated)
        Median:   median([255, 50, 50, ...]) = 50  (noise eliminated)

    Bilateral filter (edge-preserving blur):
        Blurs ONLY pixels that are BOTH spatially close AND similar in colour.
        Produces beautifully smooth results while preserving sharp edges.
        Used in portrait smoothing, depth map refinement.
        bilateral = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        # d:          diameter of pixel neighbourhood (try 9–15)
        # sigmaColor: filter sigma in colour space (try 75)
        # sigmaSpace: filter sigma in spatial space (try 75)
        Expensive: O(d² × H × W) vs O(k² × H × W) for Gaussian.

### Sharpening: Amplifying High Frequencies

    Sharpening emphasises edges and fine detail. Two equivalent approaches:

    1. Unsharp masking (the standard approach):
           sharp = original + λ × (original - blurred)
           sharp = cv2.addWeighted(original, 1 + λ, blurred, -λ, 0)

    2. Sharpening kernel:
           [[ 0, -1,  0],
            [-1,  5, -1],    # centre weight=5, neighbours=-1
            [ 0, -1,  0]]    # net effect: amplify centre vs average

    The mathematical insight:
        The Laplacian of an image measures curvature = high at edges.
        Adding the Laplacian to the original amplifies those edges:
        sharp = image + λ × Laplacian(image)

### The Frequency Domain Perspective

    Any image can be decomposed into a sum of sinusoidal waves (Fourier theory):
        Low frequencies:  slow intensity variations = background, gradients
        High frequencies: rapid intensity changes = edges, texture, noise

    Blur = low-pass filter (removes high frequencies = edges, noise)
    Sharpen = high-pass filter (amplifies high frequencies = edges)
    Edge detection = band-pass or high-pass filter

    This perspective explains WHY blurring before edge detection works:
        Noise lives in the high frequencies.
        Blurring removes noise (high freq).
        Then edge detection finds the remaining high frequencies (real edges).
        This is exactly what the Canny detector does.


##### PART 3 — EDGE DETECTION: FINDING BOUNDARIES

### What Is an Edge?

    An edge is a location of rapid intensity change in an image.
    Edges correspond to: object boundaries, surface markings, depth
    discontinuities, and lighting changes.

    The gradient of an image measures the RATE of intensity change:
        Gradient vector: ∇I = (∂I/∂x, ∂I/∂y)
        Gradient magnitude: |∇I| = √((∂I/∂x)² + (∂I/∂y)²)
        Gradient direction: θ = arctan(∂I/∂y / ∂I/∂x)

    Strong edges → large gradient magnitude.
    The direction indicates the edge orientation (perpendicular to the edge).

### Sobel Operator: Finite Difference Gradients

    The Sobel kernels approximate the image gradient:

        Kx = [[-1, 0, 1],     Ky = [[-1, -2, -1],
              [-2, 0, 2],           [ 0,  0,  0],
              [-1, 0, 1]]           [ 1,  2,  1]]

    Kx detects vertical edges (changes along x direction).
    Ky detects horizontal edges (changes along y direction).
    The central row/column weights of 2 provide smoothing to reduce noise.

        Gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)  # x-gradient
        Gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)  # y-gradient
        magnitude = np.sqrt(Gx**2 + Gy**2)
        direction = np.arctan2(Gy, Gx)                    # edge direction

    Important: use cv2.CV_64F (float64), not uint8.
    Negative gradients (going from bright to dark) are clipped to 0 with uint8.
    Taking abs() after computation recovers both edge polarities.

### Laplacian: Second-Order Derivative

    The Laplacian is the SUM of second derivatives:
        ∇²I = ∂²I/∂x² + ∂²I/∂y²

    Discrete Laplacian kernel:
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]]

    It measures CURVATURE rather than slope.
    Zero crossings of the Laplacian indicate edge locations.
    Highly sensitive to noise → always blur before applying.

        laplacian = cv2.Laplacian(grey, cv2.CV_64F)

### Canny Edge Detector: The Gold Standard

    The Canny detector (John Canny, 1986) is the most widely used edge
    detector because it optimally satisfies three criteria:
        1. Low error rate:    detects real edges, misses few
        2. Good localisation: edges marked close to true edge centres
        3. Minimal response:  one edge marker per real edge

    The Canny pipeline:
        Step 1 — Gaussian smoothing:
            Remove noise. The σ parameter controls the scale of edges detected.
            Large σ: detects coarse edges, misses fine detail.
            Small σ: detects fine detail, more sensitive to noise.

        Step 2 — Gradient computation (Sobel):
            Compute Gx, Gy, magnitude, and direction at every pixel.

        Step 3 — Non-maximum suppression (NMS):
            For each pixel, check if it is a LOCAL MAXIMUM in the gradient
            direction. If not, suppress it (set to 0).
            This "thins" edges from thick ridges to 1-pixel-wide lines.

        Step 4 — Double thresholding:
            High threshold (T_high): pixels above this are DEFINITELY edges.
            Low threshold  (T_low):  pixels below this are DEFINITELY not edges.
            Between T_low and T_high: CANDIDATE edges.

        Step 5 — Edge tracking by hysteresis:
            A candidate pixel is kept AS AN EDGE if it is connected to a
            definite edge pixel (above T_high). Isolated candidates are discarded.
            This connects broken edges and removes noise artifacts.

        edges = cv2.Canny(grey, threshold1=50, threshold2=150)
        # threshold1: low threshold (T_low)
        # threshold2: high threshold (T_high)
        # Rule of thumb: T_high = 2×T_low to 3×T_low

        Tuning Canny:
            Too many edges:  increase both thresholds
            Missing edges:   decrease both thresholds
            Noisy edges:     blur more before Canny
            Broken edges:    decrease low threshold
            Thick edges:     should not happen (NMS handles this)

### Gradient Orientation and Edge Types

    The gradient direction θ is PERPENDICULAR to the edge direction.
    Classifying edges by gradient angle (binned into 4 directions):
        0°   (±22.5°):  vertical edge    (|)
        45°  (±22.5°):  diagonal edge    (/)
        90°  (±22.5°):  horizontal edge  (—)
        135° (±22.5°):  diagonal edge    (\\)

    This binning is used in HOG (Histogram of Oriented Gradients)
    descriptors and in Canny's NMS step.


##### PART 4 — MORPHOLOGICAL OPERATIONS AND THRESHOLDING

### Binary Images and Thresholding

    Thresholding converts a greyscale image to binary (black/white):
        pixel > threshold → 255 (white = foreground)
        pixel ≤ threshold → 0   (black = background)

    Simple threshold:
        _, binary = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)
        # Returns the threshold used and the binary image

    Otsu's method — automatic threshold selection:
        Otsu's algorithm finds the threshold that minimises the weighted
        variance of the two pixel classes (background vs foreground).
        It assumes the histogram is bimodal (has two peaks).
        thresh, binary = cv2.threshold(grey, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Adaptive thresholding — for uneven illumination:
        Computes a local threshold for each pixel based on its neighbourhood.
        Handles images where lighting varies across the scene.
        binary = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or MEAN_C
            cv2.THRESH_BINARY,
            blockSize = 11,    # neighbourhood size (must be odd)
            C         = 2,     # subtract this constant from mean/weighted mean
        )

    Common pitfall: Otsu fails on multi-modal histograms (more than 2 classes).
    Use k-means or GMM-based segmentation for those cases.

### Morphological Operations: Shape Processing

    Morphological operations process binary images based on a structuring
    element (SE) — a small shape that probes the image.

    The two primitives:

    EROSION:
        Keeps a foreground pixel ONLY if the entire SE fits within the foreground.
        Effect: shrinks objects, removes thin protrusions, disconnects objects.
        eroded = cv2.erode(binary, kernel, iterations=1)

    DILATION:
        Sets a pixel to foreground if ANY part of the SE touches foreground.
        Effect: grows objects, fills thin gaps, connects nearby objects.
        dilated = cv2.dilate(binary, kernel, iterations=1)

    OPENING = Erosion then Dilation:
        Removes small objects and thin protrusions without significantly
        changing the size of larger objects.
        Use for: removing noise while preserving shape.
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    CLOSING = Dilation then Erosion:
        Fills small holes and gaps within objects without significantly
        changing the size.
        Use for: filling gaps, connecting nearby parts of the same object.
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    GRADIENT = Dilation − Erosion:
        Extracts the boundary of objects (outline).
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    TOP HAT = Original − Opening:
        Extracts small bright features from a dark background.
        tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)

    BLACK HAT = Closing − Original:
        Extracts small dark features from a bright background.
        blackhat = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)

    Structuring elements:
        rect_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,    (5,5))
        ellipse_se   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cross_se     = cv2.getStructuringElement(cv2.MORPH_CROSS,   (5,5))


##### PART 5 — CONTOURS, SHAPE ANALYSIS, AND CONNECTED COMPONENTS

### Contour Detection

    A contour is a curve joining all continuous points along a boundary with
    the same intensity. In binary images, contours are the boundaries of
    white (foreground) regions.

        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,   # retrieval mode (see below)
            cv2.CHAIN_APPROX_SIMPLE  # compression method
        )

    Retrieval modes:
        RETR_EXTERNAL:    only outermost contours (no holes)
        RETR_LIST:        all contours, no hierarchy
        RETR_CCOMP:       two-level hierarchy (outer + holes)
        RETR_TREE:        full hierarchy tree

    Approximation methods:
        CHAIN_APPROX_NONE:   all contour points (verbose)
        CHAIN_APPROX_SIMPLE: compress runs (e.g. a rectangle = 4 points)

### Contour Descriptors

    Area:
        area = cv2.contourArea(contour)

    Perimeter (arc length):
        perimeter = cv2.arcLength(contour, closed=True)

    Centroid (via image moments):
        M  = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])   # centroid x
        cy = int(M['m01'] / M['m00'])   # centroid y

    Bounding rectangle:
        x, y, w, h = cv2.boundingRect(contour)      # axis-aligned
        rect = cv2.minAreaRect(contour)              # rotated (min area)
        box  = cv2.boxPoints(rect)                   # 4 corners of rotated rect

    Minimum enclosing circle:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

    Convex hull:
        hull = cv2.convexHull(contour)
        isConvex = cv2.isContourConvex(contour)

    Shape descriptors:
        Circularity = 4π × Area / Perimeter²    (1.0 = perfect circle)
        Aspect ratio = Width / Height            (1.0 = square)
        Extent = Area / (W × H bounding rect)   (fill ratio)
        Solidity = Area / Convex hull area       (convexity measure)

    Polygon approximation:
        epsilon  = 0.02 * cv2.arcLength(contour, True)
        approx   = cv2.approxPolyDP(contour, epsilon, True)
        # epsilon: max distance from contour to approximation
        # len(approx)==3 → triangle; ==4 → quadrilateral; etc.

### Connected Components: Labelled Regions

    Connected components analysis labels every connected region in a binary
    image with a unique integer ID:

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

    stats[label] = [x, y, width, height, area]
    labels[y, x] = integer label (0 = background)

    Filtering objects by size (removing noise):
        for label in range(1, num_labels):          # skip background (0)
            if stats[label, cv2.CC_STAT_AREA] < 50:
                binary[labels == label] = 0         # remove small blobs


##### PART 6 — FEATURE DETECTION AND DESCRIPTION

### Why Features?

    Matching two images of the same scene requires finding corresponding
    points. Matching every pixel is computationally intractable and fragile.
    Instead, we find KEYPOINTS — distinctive locations with repeatable
    properties — and describe each with a compact DESCRIPTOR.

    A good feature detector finds points that are:
        REPEATABLE:     detected at the same location across images
        DISTINCTIVE:    describable so they can be matched
        INVARIANT:      stable under rotation, scale, illumination changes
        EFFICIENT:      fast to compute

### Harris Corner Detector

    A corner is a point where intensity changes significantly in MULTIPLE
    directions. Unlike edges (one direction of change) or flat regions
    (no direction of change), corners are uniquely localisable.

    The Harris measure analyzes the second moment matrix (structure tensor):
        M = Σ[∂I/∂x², ∂I/∂x·∂I/∂y; ∂I/∂x·∂I/∂y, ∂I/∂y²]

    Eigenvalues of M describe the neighbourhood:
        Both small:   flat region (uniform intensity)
        One large:    edge (strong gradient in one direction)
        Both large:   corner (strong gradient in all directions)

    Harris response: R = det(M) − k·trace(M)²
        R >> 0:  corner
        R << 0:  edge
        R ≈ 0:   flat

    In OpenCV:
        dst = cv2.cornerHarris(grey, blockSize=2, ksize=3, k=0.04)
        # blockSize: neighbourhood size for covariance matrix
        # ksize: Sobel aperture parameter
        # k: Harris detector sensitivity (0.04–0.06 typical)
        corners = image.copy()
        corners[dst > 0.01 * dst.max()] = [0, 0, 255]  # mark corners red

### SIFT: Scale-Invariant Feature Transform

    SIFT (Lowe, 2004) is the landmark feature detector/descriptor.
    It detects and describes features that are invariant to scale,
    rotation, and partially invariant to illumination and viewpoint.

    SIFT detection pipeline:
        1. Scale-space extrema: build a Gaussian pyramid at multiple scales,
           compute Difference of Gaussians (DoG = two consecutive Gaussians
           subtracted). Find local extrema in space AND scale.
        2. Keypoint localisation: reject low-contrast points and edge
           responses (using Hessian matrix).
        3. Orientation assignment: compute gradient histogram in the
           neighbourhood; dominant orientation assigned → rotation invariance.
        4. Descriptor computation: 4×4 grid of 8-bin gradient histograms
           = 128-dimensional descriptor. Normalised for illumination invariance.

    The 128-dimensional SIFT descriptor is a "fingerprint" of the local
    patch that is robust to many image transformations.

    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(grey, None)

### ORB: Oriented FAST and Rotated BRIEF

    ORB is a free, fast alternative to SIFT/SURF designed by Rublee et al. (2011).
    It is the default feature detector for real-time applications.

    ORB = FAST keypoint detector + BRIEF descriptor + orientation normalisation

    FAST (Features from Accelerated Segment Test):
        A pixel is a corner if ≥9 consecutive pixels in a circle of 16 are
        all brighter or all darker than the centre by a threshold.
        Machine-learned decision tree makes this extremely fast.

    BRIEF (Binary Robust Independent Elementary Features):
        Compares 256 random pixel pairs in the neighbourhood.
        Each comparison: 0 or 1. Result: 256-bit binary string.
        Matching: Hamming distance (XOR + popcount) — extremely fast.

    ORB adds rotation invariance by computing BRIEF relative to the
    keypoint's dominant gradient orientation.

    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(grey, None)

### Feature Matching

    Brute-force matcher (exact):
        bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # for ORB
        bf    = cv2.BFMatcher(cv2.NORM_L2,      crossCheck=True)  # for SIFT
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda m: m.distance)

    FLANN matcher (approximate, faster for large descriptor sets):
        Optimised for high-dimensional descriptors (SIFT/SURF).
        FLANN_INDEX_KDTREE=1 for float descriptors.
        FLANN_INDEX_LSH=6 for binary descriptors (ORB).

    Ratio test (Lowe's ratio test — best practice):
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        # Keep only if best match is significantly better than second best.
        # Rejects ambiguous matches that could correspond to multiple keypoints.

### Homography: Geometric Image Alignment

    Given 4+ matching point pairs, RANSAC finds the 3×3 homography matrix
    that maps one image plane to another:

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=5.0)

    RANSAC (Random Sample Consensus):
        Iteratively samples minimal point sets, fits a model, counts inliers.
        Robust to outlier matches (mismatches) — critical for feature matching.

    Applications:
        Panorama stitching, document scanning (dewarping), augmented reality,
        image rectification, aerial image alignment.

        warped = cv2.warpPerspective(img1, H, (width, height))


##### PART 7 — GEOMETRIC TRANSFORMS AND SPATIAL OPERATIONS

### Affine Transforms

    An affine transform preserves parallel lines and ratios of distances.
    It can represent: translation, rotation, scaling, shearing, and any
    combination. Requires 3 point correspondences to compute (6 DOF).

        M = cv2.getAffineTransform(src_3pts, dst_3pts)   # from 3 points
        # or:
        M = cv2.getRotationMatrix2D(centre, angle, scale)   # rotation + scale

        transformed = cv2.warpAffine(image, M, (width, height))

    Decomposing a rotation matrix:
        angle = np.degrees(np.arctan2(M[1,0], M[0,0]))

### Perspective Transform (Homography)

    A perspective transform (projective transform) maps any quadrilateral
    to any other quadrilateral. Represents projections of 3D planes onto 2D.
    Used for: document scanner, bird's-eye view, AR plane detection.

        M = cv2.getPerspectiveTransform(src_4pts, dst_4pts)
        warped = cv2.warpPerspective(image, M, (width, height))

    Document scanning (4-point transform):
        1. Detect document corners (contour finding or corner detection)
        2. Order corners: top-left, top-right, bottom-right, bottom-left
        3. Compute output width/height from maximum side lengths
        4. Apply getPerspectiveTransform to correct the perspective

### Remapping and Lens Distortion

    cv2.remap() applies arbitrary pixel-to-pixel mappings:
        For every output pixel (x, y), specify the source pixel (map_x[y,x], map_y[y,x]).

    Camera lens distortion correction:
        Real lenses exhibit radial distortion (barrel/pincushion) and
        tangential distortion (lens not perfectly parallel to sensor).
        Camera calibration with a chessboard estimates the intrinsic matrix
        K and distortion coefficients D.
        undistorted = cv2.undistort(image, K, D)

    Camera calibration pipeline:
        objpoints, imgpoints = [], []
        for frame in chessboard_images:
            ret, corners = cv2.findChessboardCorners(grey, (9,6), None)
            if ret:
                objpoints.append(objp)         # 3D world points
                imgpoints.append(corners)      # 2D image points
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, grey.shape[::-1], None, None
        )

### Resize, Rotate, and Flip

    Resize:
        resized = cv2.resize(image, (new_w, new_h),
                              interpolation=cv2.INTER_LINEAR)
        # Interpolation methods:
        # INTER_NEAREST:  fastest, blocky (for integer upscaling)
        # INTER_LINEAR:   bilinear (default, good quality)
        # INTER_CUBIC:    bicubic (better quality, slower)
        # INTER_LANCZOS4: best quality (for downscaling only)
        # INTER_AREA:     best for shrinking (avoids aliasing)

    Rotate without cropping:
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        # Adjust for new bounding box to avoid cropping:
        cos, sin  = abs(M[0,0]), abs(M[0,1])
        new_w     = int(h*sin + w*cos)
        new_h     = int(h*cos + w*sin)
        M[0, 2]  += (new_w - w) / 2
        M[1, 2]  += (new_h - h) / 2
        rotated   = cv2.warpAffine(image, M, (new_w, new_h))

    Flip:
        flipped_h = cv2.flip(image, 1)   # horizontal flip
        flipped_v = cv2.flip(image, 0)   # vertical flip
        flipped_b = cv2.flip(image, -1)  # both axes


##### PART 8 — OPTICAL FLOW, MOTION ANALYSIS, AND OBJECT DETECTION

### Optical Flow: Measuring Motion

    Optical flow is the apparent motion of pixels between consecutive frames.
    It assumes brightness constancy: the intensity of a moving point does
    not change between frames:
        I(x, y, t) = I(x + δx, y + δy, t + δt)

    The optical flow constraint equation (from Taylor expansion):
        Ix·u + Iy·v + It = 0
    where Ix, Iy = spatial gradients, It = temporal gradient, (u,v) = flow.

    This one equation has two unknowns — the aperture problem.
    Different algorithms resolve this ambiguity differently.

    Lucas-Kanade (sparse optical flow):
        Assumes constant flow within a small neighbourhood.
        Computes flow at KEYPOINTS only (sparse).
        Fast and accurate for tracking specific features.

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            grey_prev, grey_curr, p0, None,
            winSize=(15,15),       # search window size
            maxLevel=2,            # pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # p0:   tracked points from previous frame (Nx1x2 float32)
        # p1:   corresponding points in current frame
        # st:   status: 1=found, 0=not found
        # err:  per-point tracking error

    Farneback (dense optical flow):
        Computes flow at EVERY pixel (dense).
        Approximates local image patch by polynomial expansion.
        Returns a 2-channel array: flow[y,x] = (u, v) for every pixel.

        flow = cv2.calcOpticalFlowFarneback(
            grey_prev, grey_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Visualise with HSV: H=direction, S=255, V=magnitude

### Background Subtraction

    Separating moving objects (foreground) from the static scene (background).

    MOG2 (Mixture of Gaussians):
        Models each pixel as a mixture of Gaussians that evolve over time.
        Automatically adapts to gradual scene changes.
        Handles multi-modal backgrounds (e.g. waving branches).

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,         # frames for background model
            varThreshold=16,     # threshold for background/foreground decision
            detectShadows=True,  # mark shadows grey (127) instead of white (255)
        )
        fg_mask = mog2.apply(frame)   # call each frame; model updates

    KNN (K-Nearest Neighbours):
        Non-parametric background model. More accurate than MOG2 for
        complex backgrounds. Slower.

        knn = cv2.createBackgroundSubtractorKNN()
        fg_mask = knn.apply(frame)

### Template Matching

    Slide a template image over the source and compute a similarity score
    at each position. The highest score indicates the best match location.

        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left  = max_loc   # for TM_CCOEFF_NORMED and TM_CCORR_NORMED
        bottom_right = (top_left[0] + tw, top_left[1] + th)

    Methods:
        TM_SQDIFF:       sum of squared differences (min=best)
        TM_CCORR_NORMED: normalised cross-correlation (max=best)
        TM_CCOEFF_NORMED: normalised correlation coefficient (max=best, robust)

    Limitations: sensitive to scale, rotation, lighting changes.
    For robust matching use feature-based matching instead.

### HOG + SVM: Classical Object Detection

    HOG (Histogram of Oriented Gradients, Dalal & Triggs 2005):
        Divides the image into cells (8×8 pixels).
        For each cell, computes an 8-bin gradient orientation histogram.
        Normalises blocks of 2×2 cells for illumination invariance.
        Concatenates all histograms → a fixed-length feature vector.

    HOG + SVM pipeline (pedestrian detection):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(
            image,
            winStride   = (4, 4),
            padding     = (8, 8),
            scale       = 1.05,    # pyramid scale factor
        )

    Why HOG works: gradient orientations are locally consistent at object
    boundaries. A standing person has a characteristic HOG signature.
    HOG is invariant to local illumination changes (normalisation).

### The Classical-to-Deep Learning Bridge

    Deep CNNs learned to outperform HOG features for detection after AlexNet (2012).
    But the relationship is complementary, not replacement:

        Classical CV role in deep learning pipelines:
            Preprocessing:    resize, normalise, colour convert
            Data augmentation: flip, rotate, crop, adjust brightness
            Post-processing:  NMS (non-maximum suppression) on detection boxes
            Visualisation:    drawing bounding boxes, contours on outputs
            Calibration:      camera calibration for 3D vision
            Speed:            classical features for real-time auxiliary tasks

    OpenCV + Deep Learning (DNN module):
        net = cv2.dnn.readNet("yolov5.onnx")
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640,640),
                                       swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

    OpenCV's DNN module runs ONNX, TensorFlow, PyTorch, Caffe models
    entirely in C++ — no Python ML dependency required at inference time.
    Supports CPU, GPU (CUDA), OpenCL, Myriad X (Intel Neural Compute Stick).

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Image Fundamentals — Pixels, Colour Spaces, and Histograms": {
        "description": (
            "Core image manipulation and analysis with NumPy and OpenCV. "
            "Synthetic image generation without loading files. "
            "Pixel access, channel splitting, and colour arithmetic. "
            "Colour space conversions: BGR → HSV → LAB → YCrCb. "
            "HSV colour detection with inRange() masking. "
            "Histogram computation for each channel. "
            "Histogram equalisation and CLAHE adaptive contrast. "
            "Image blending, alpha compositing, and bitwise operations. "
            "Region of interest (ROI) extraction and copy. "
            "Pixel intensity statistics: mean, std, min, max per channel. "
            "Colour constancy: white balance estimation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import cv2
import time

print("=" * 65)
print("  IMAGE FUNDAMENTALS — PIXELS, COLOUR SPACES, HISTOGRAMS")
print("=" * 65)
print()
print(f"  OpenCV version: {cv2.__version__}")
print(f"  NumPy  version: {np.__version__}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Synthetic image creation (no file loading needed)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Synthetic image creation and pixel anatomy")
print("━" * 65)
print()

def make_test_image(h=256, w=256):
    """
    Create a rich synthetic test image with:
    - Coloured quadrants (BGR)
    - A grey gradient strip
    - A bright circular region
    - Random noise overlay
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mid_h, mid_w = h // 2, w // 2

    # Four coloured quadrants
    img[:mid_h, :mid_w]   = [255,  50,  50]   # top-left:     Blue-ish
    img[:mid_h, mid_w:]   = [ 50, 200,  50]   # top-right:    Green-ish
    img[mid_h:, :mid_w]   = [ 50,  50, 220]   # bottom-left:  Red-ish
    img[mid_h:, mid_w:]   = [200, 180,  50]   # bottom-right: Cyan-ish

    # Diagonal gradient strip
    for i in range(w):
        val = int(255 * i / w)
        cv2.line(img, (i, 0), (i, 15), (val, val, val), 1)

    # White circle in the centre
    cv2.circle(img, (w//2, h//2), radius=30, color=(255, 255, 255), thickness=-1)

    # Add mild Gaussian noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

img = make_test_image()
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"  Image shape:   {img.shape}  (H, W, C)")
print(f"  Image dtype:   {img.dtype}")
print(f"  Total pixels:  {img.shape[0] * img.shape[1]:,}")
print(f"  Memory:        {img.nbytes / 1024:.1f} KB")
print()

# Pixel access
px = img[64, 64]      # pixel at row=64, col=64 → (y=64, x=64)
print(f"  Pixel at (row=64, col=64)  →  BGR = {px}")
print(f"  Pixel at (row=64, col=192) →  BGR = {img[64, 192]}")
print()

# Channel splitting
B, G, R = cv2.split(img)
print(f"  Channel statistics (mean ± std):")
print(f"    Blue:  {B.mean():.1f} ± {B.std():.1f}   range [{B.min()}, {B.max()}]")
print(f"    Green: {G.mean():.1f} ± {G.std():.1f}   range [{G.min()}, {G.max()}]")
print(f"    Red:   {R.mean():.1f} ± {R.std():.1f}   range [{R.min()}, {R.max()}]")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Colour space conversions
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Colour space conversions: HSV, LAB, YCrCb")
print("━" * 65)
print()

hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

print(f"  Colour space channel ranges (sampled at 64,64):")
print(f"  {'Space':<8} {'Ch1 (name)':>12} {'Ch2 (name)':>12} {'Ch3 (name)':>12}")
print(f"  {'─'*50}")

spaces = [
    ("BGR",   img,   "Blue",  "Green", "Red"),
    ("HSV",   hsv,   "Hue",   "Sat",   "Val"),
    ("LAB",   lab,   "L*",    "A*",    "B*"),
    ("YCrCb", ycrcb, "Y",     "Cr",    "Cb"),
]
for name, arr, c1, c2, c3 in spaces:
    v1, v2, v3 = arr[64, 64]
    print(f"  {name:<8} {f'{c1}={v1}':>12} {f'{c2}={v2}':>12} {f'{c3}={v3}':>12}")
print()

# Demonstrate why HSV is better for colour detection
print(f"  HSV colour detection demonstration:")
print(f"  Detecting 'blue-ish' region (top-left quadrant):")

# BGR approach: fragile, depends on exact values
bgr_lower = np.array([200,  20,  20])
bgr_upper = np.array([255,  80,  80])
bgr_mask  = cv2.inRange(img, bgr_lower, bgr_upper)
bgr_hits  = cv2.countNonZero(bgr_mask)

# HSV approach: robust, based on hue
# Blue hue is approx H=100-130 in OpenCV (0-179 range)
hsv_lower = np.array([100, 80, 80])
hsv_upper = np.array([130, 255, 255])
hsv_mask  = cv2.inRange(hsv, hsv_lower, hsv_upper)
hsv_hits  = cv2.countNonZero(hsv_mask)

print(f"    BGR mask detected pixels: {bgr_hits:,}")
print(f"    HSV mask detected pixels: {hsv_hits:,}")
print(f"    Expected (quarter of image): ~{256*256//4:,}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Histograms and equalisation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Histograms and contrast enhancement")
print("━" * 65)
print()

def histogram_stats(channel_arr, name):
    hist = cv2.calcHist([channel_arr], [0], None, [256], [0, 256])
    hist = hist.flatten()
    n_pixels = channel_arr.size
    # Find most common intensity bin
    peak_val = hist.argmax()
    peak_pct = 100 * hist[peak_val] / n_pixels
    # Compute entropy (distribution evenness)
    p = hist / n_pixels + 1e-10
    entropy = -np.sum(p * np.log2(p))
    return peak_val, peak_pct, entropy

print(f"  Histogram analysis:")
print(f"  {'Channel':<10} {'Peak intensity':>16} {'Peak %':>8} {'Entropy (bits)':>16}")
print(f"  {'─'*54}")
for ch_name, ch_arr in [("Blue", B), ("Green", G), ("Red", R), ("Grey", grey)]:
    peak_v, peak_p, ent = histogram_stats(ch_arr, ch_name)
    print(f"  {ch_name:<10} {peak_v:>16} {peak_p:>8.2f}% {ent:>16.3f}")
print()

# Create a dark (underexposed) image for equalisation demo
dark_img  = (grey * 0.3).astype(np.uint8)
print(f"  Dark image stats: mean={dark_img.mean():.1f}, std={dark_img.std():.1f}")

# Global histogram equalisation
equal_global = cv2.equalizeHist(dark_img)
print(f"  After equalizeHist: mean={equal_global.mean():.1f}, std={equal_global.std():.1f}")

# CLAHE
clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equal_clahe = clahe.apply(dark_img)
print(f"  After CLAHE:       mean={equal_clahe.mean():.1f}, std={equal_clahe.std():.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Image arithmetic and blending
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Image arithmetic: blending, masking, bitwise ops")
print("━" * 65)
print()

img2 = make_test_image()   # second test image
np.random.shuffle(img2.reshape(-1, 3))   # shuffle colours

# Alpha blending
alpha   = 0.6
blended = cv2.addWeighted(img, alpha, img2, 1 - alpha, gamma=0)
print(f"  Alpha blend (α={alpha}): dtype={blended.dtype}, "
      f"mean={blended.mean():.1f}")

# Arithmetic operations (with overflow protection)
brighter = cv2.add(img, np.array([30, 30, 30], dtype=np.uint8))   # clamps at 255
darker   = cv2.subtract(img, np.array([30, 30, 30], dtype=np.uint8))  # clamps at 0
print(f"  cv2.add(img, 30):      mean {img.mean():.1f} → {brighter.mean():.1f}  (saturates at 255)")
print(f"  cv2.subtract(img, 30): mean {img.mean():.1f} → {darker.mean():.1f}  (saturates at 0)")
print()

# Mask-based painting
mask = np.zeros(grey.shape, dtype=np.uint8)
cv2.circle(mask, (128, 128), 40, 255, -1)    # circular mask

masked_img  = img.copy()
masked_img[mask == 255] = [0, 255, 255]       # paint selected region yellow
print(f"  Masked region painting: painted {(mask == 255).sum():,} pixels yellow")

# Bitwise operations
circle_img  = np.zeros_like(img)
cv2.circle(circle_img, (128, 128), 50, (255, 255, 255), -1)
and_result  = cv2.bitwise_and(img, circle_img)
print(f"  bitwise_and (extract circular region): non-zero pixels = {np.count_nonzero(and_result[:,:,0]):,}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: ROI operations and copy-paste
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — ROI: region of interest extraction and manipulation")
print("━" * 65)
print()

# Extract and analyse a region of interest
roi = img[50:150, 50:150]   # rows 50-150, cols 50-150
print(f"  ROI shape: {roi.shape}  (100×100 pixels from top-left region)")
print(f"  ROI mean BGR: {roi.mean(axis=(0,1)).round(1)}")

# Paste ROI into another location
output = img.copy()
output[150:250, 150:250] = roi   # paste at new location
diff_pixels = np.count_nonzero(output[150:250, 150:250] != img[150:250, 150:250])
print(f"  Pasted ROI to [150:250, 150:250]: {diff_pixels:,} pixels changed")
print()

# Border padding
pad_constant = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
pad_reflect  = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                                   cv2.BORDER_REFLECT)
pad_replicate = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                                    cv2.BORDER_REPLICATE)
print(f"  Border padding (10px on each side):")
print(f"    CONSTANT (black):   {img.shape[:2]} → {pad_constant.shape[:2]}")
print(f"    REFLECT:            {img.shape[:2]} → {pad_reflect.shape[:2]}")
print(f"    REPLICATE:          {img.shape[:2]} → {pad_replicate.shape[:2]}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: White balance and colour statistics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — White balance and perceptual colour analysis")
print("━" * 65)
print()

# Grey-world white balance assumption: average scene is neutral grey
mean_b = float(np.mean(B))
mean_g = float(np.mean(G))
mean_r = float(np.mean(R))
mean_grey_world = (mean_b + mean_g + mean_r) / 3

gain_b = mean_grey_world / (mean_b + 1e-6)
gain_g = mean_grey_world / (mean_g + 1e-6)
gain_r = mean_grey_world / (mean_r + 1e-6)

balanced = np.zeros_like(img)
balanced[:,:,0] = np.clip(B * gain_b, 0, 255).astype(np.uint8)
balanced[:,:,1] = np.clip(G * gain_g, 0, 255).astype(np.uint8)
balanced[:,:,2] = np.clip(R * gain_r, 0, 255).astype(np.uint8)

B2, G2, R2 = cv2.split(balanced)
print(f"  Grey-world white balance:")
print(f"  {'Channel':<8} {'Before mean':>14} {'Gain':>8} {'After mean':>12}")
print(f"  {'─'*46}")
for ch, bef, gain, aft in [("Blue",  mean_b, gain_b, float(B2.mean())),
                             ("Green", mean_g, gain_g, float(G2.mean())),
                             ("Red",   mean_r, gain_r, float(R2.mean()))]:
    print(f"  {ch:<8} {bef:>14.2f} {gain:>8.3f} {aft:>12.2f}")
print()

# Perceptual colour distance in LAB space
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
p1_lab  = lab_img[64, 64]     # blue-ish quadrant
p2_lab  = lab_img[64, 192]    # green-ish quadrant
p3_lab  = lab_img[192, 64]    # red-ish quadrant

delta_E_12 = np.sqrt(np.sum((p1_lab - p2_lab)**2))
delta_E_13 = np.sqrt(np.sum((p1_lab - p3_lab)**2))
delta_E_23 = np.sqrt(np.sum((p2_lab - p3_lab)**2))

print(f"  CIE76 colour distance (ΔE) in LAB space:")
print(f"    Blue  ↔ Green: ΔE = {delta_E_12:.2f}")
print(f"    Blue  ↔ Red:   ΔE = {delta_E_13:.2f}")
print(f"    Green ↔ Red:   ΔE = {delta_E_23:.2f}")
print(f"  (ΔE > 2.3 is visually perceptible; > 10 is a dramatic difference)")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · Filtering, Edge Detection & Morphology — Signal Processing": {
        "description": (
            "Spatial filtering and edge detection from first principles. "
            "Gaussian, median, and bilateral blur comparison. "
            "Custom kernel convolution with cv2.filter2D. "
            "Frequency perspective: low-pass vs high-pass analysis. "
            "Sobel gradient: magnitude, direction, and phase visualisation. "
            "Laplacian second-order derivative. "
            "Canny edge detector: pipeline step-by-step. "
            "Threshold comparison: simple, Otsu, and adaptive. "
            "Morphological operations: erosion, dilation, opening, closing. "
            "Structuring element shape comparison. "
            "Connected components analysis with area filtering. "
            "Morphological gradient for boundary extraction."
        ),
        "language": "python",
        "code": '''
import numpy as np
import cv2
import time

print("=" * 65)
print("  FILTERING, EDGE DETECTION & MORPHOLOGY")
print("=" * 65)
print()

# ── Synthetic test images ─────────────────────────────────────────────
rng = np.random.default_rng(42)

def make_gradient_image(h=256, w=256):
    """Clean gradient + geometric shapes for filter demo."""
    img = np.zeros((h, w), dtype=np.uint8)
    # Gradient background
    for col in range(w):
        img[:, col] = int(255 * col / w)
    # Shapes
    cv2.rectangle(img, (30, 30), (90, 90), 200, -1)
    cv2.circle(img, (160, 80), 40, 50, -1)
    cv2.ellipse(img, (200, 180), (50, 30), 45, 0, 360, 180, -1)
    # Add thin lines (edge detection targets)
    cv2.line(img, (10, 10), (246, 246), 255, 2)
    cv2.line(img, (10, 246), (246, 10), 255, 2)
    return img

def add_noise(img, sigma=20, salt_pct=0.02):
    """Add Gaussian noise and salt-and-pepper noise."""
    noisy = img.astype(np.float32)
    noisy += rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy  = np.clip(noisy, 0, 255).astype(np.uint8)
    n_sp   = int(img.size * salt_pct)
    sp_idx = rng.integers(0, img.size, n_sp)
    noisy.flat[sp_idx[:n_sp//2]] = 255   # salt
    noisy.flat[sp_idx[n_sp//2:]] = 0     # pepper
    return noisy

clean = make_gradient_image()
noisy = add_noise(clean, sigma=20, salt_pct=0.02)

print(f"  Test image: {clean.shape}, dtype={clean.dtype}")
print(f"  Clean image SNR:  {clean.mean():.1f} ± {clean.std():.1f}")
print(f"  Noisy image SNR:  {noisy.mean():.1f} ± {noisy.std():.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Blur filters comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Blur filters: Gaussian vs Median vs Bilateral")
print("━" * 65)
print()

def noise_reduction_quality(original, filtered):
    """PSNR: Peak Signal-to-Noise Ratio (higher = better denoising)."""
    mse = float(np.mean((original.astype(float) - filtered.astype(float))**2))
    if mse < 1e-10: return float('inf')
    return 10 * np.log10(255**2 / mse)

# Edge preservation: standard deviation near an edge
def edge_preservation(filtered, edge_y=64, edge_x=128, window=5):
    """Measure sharpness at an edge: higher std = sharper edge."""
    region = filtered[edge_y-window:edge_y+window, edge_x-window:edge_x+window]
    return float(region.std())

filters_to_test = {
    "Original":          clean,
    "Gaussian 3×3":      cv2.GaussianBlur(noisy, (3, 3), 0),
    "Gaussian 7×7":      cv2.GaussianBlur(noisy, (7, 7), 0),
    "Median 3×3":        cv2.medianBlur(noisy, 3),
    "Median 7×7":        cv2.medianBlur(noisy, 7),
    "Bilateral":         cv2.bilateralFilter(noisy, 9, 75, 75),
}

print(f"  {'Filter':<20} {'PSNR (dB)':>12} {'Edge sharpness':>16} {'Time (ms)':>12}")
print(f"  {'─'*64}")
for name, filtered in filters_to_test.items():
    t0   = time.perf_counter()
    psnr = noise_reduction_quality(clean, filtered)
    ep   = edge_preservation(filtered)
    t_ms = (time.perf_counter() - t0) * 1000
    psnr_s = f"{psnr:.2f}" if psnr != float('inf') else "  inf  "
    print(f"  {name:<20} {psnr_s:>12} {ep:>16.3f} {t_ms:>12.3f}")

print()
print(f"  Interpretation:")
print(f"    PSNR higher → better noise removal")
print(f"    Edge sharpness higher → edges better preserved")
print(f"    Bilateral: best edge preservation, high PSNR, but slowest")
print(f"    Median: best for salt-and-pepper noise, excellent PSNR")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom kernels with filter2D
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Custom kernels: sharpen, emboss, Laplacian")
print("━" * 65)
print()

kernels = {
    "Identity":    np.array([[0,0,0],[0,1,0],[0,0,0]], np.float32),
    "Sharpen":     np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32),
    "Emboss":      np.array([[-2,-1,0],[-1,1,1],[0,1,2]], np.float32),
    "Box blur":    np.ones((5,5), np.float32) / 25,
    "Laplacian":   np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32),
    "Edge horiz":  np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], np.float32),
    "Edge vert":   np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], np.float32),
}

print(f"  Custom kernel results on clean image (mean pixel value):")
print(f"  {'Kernel':<14} {'Output mean':>14} {'Output std':>12} {'Description'}")
print(f"  {'─'*65}")
for name, k in kernels.items():
    result = cv2.filter2D(clean, -1, k)
    desc   = ("no change" if name=="Identity" else
              "amplify edges" if name=="Sharpen" else
              "3D relief effect" if name=="Emboss" else
              "smooth, soften" if "blur" in name else
              "curvature/edges" if name=="Laplacian" else "gradient")
    print(f"  {name:<14} {result.mean():>14.2f} {result.std():>12.2f} {desc}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Sobel gradient analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Sobel gradients: magnitude and direction")
print("━" * 65)
print()

# Compute Sobel gradients on blurred image (noise reduction first)
blurred = cv2.GaussianBlur(clean, (5, 5), 0)
Gx  = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
Gy  = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
Gmag = np.sqrt(Gx**2 + Gy**2)
Gdir = np.degrees(np.arctan2(Gy, Gx))   # direction in degrees

print(f"  Sobel gradient analysis:")
print(f"    Gx range:   [{Gx.min():.1f}, {Gx.max():.1f}]  (vertical edges)")
print(f"    Gy range:   [{Gy.min():.1f}, {Gy.max():.1f}]  (horizontal edges)")
print(f"    |G| range:  [{Gmag.min():.1f}, {Gmag.max():.1f}]  (edge strength)")
print(f"    θ range:    [{Gdir.min():.1f}°, {Gdir.max():.1f}°]  (edge direction)")
print()

# Gradient orientation histogram (HOG-style)
strong_mask  = Gmag > 50
strong_dirs  = Gdir[strong_mask]
bins         = np.linspace(-180, 180, 9)   # 8 orientation bins
hist_dirs, _ = np.histogram(strong_dirs, bins=bins)
print(f"  Gradient orientation histogram (strong edges, |G|>50):")
bin_labels = ["W", "NW", "N", "NE", "E", "SE", "S", "SW"]
for lbl, cnt in zip(bin_labels, hist_dirs):
    bar = "█" * (cnt // max(hist_dirs.max()//20, 1))
    print(f"    {lbl:>3}: {cnt:>6}  {bar}")
print()

# Edge statistics at different thresholds
print(f"  Edge pixels at different magnitude thresholds:")
for thresh in [20, 50, 100, 150, 200]:
    n_edges = (Gmag > thresh).sum()
    pct     = 100 * n_edges / Gmag.size
    print(f"    |G| > {thresh:>3}: {n_edges:>8,} pixels  ({pct:.1f}% of image)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Canny edge detector analysis
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Canny edge detector: threshold tuning")
print("━" * 65)
print()

# Test different threshold combinations
canny_configs = [
    (20,  60,  "very low (catches noise)"),
    (50,  150, "standard (recommended)"),
    (80,  240, "strict (strong edges only)"),
    (100, 300, "very strict"),
]

print(f"  {'T_low':>6} {'T_high':>7} {'Edge pixels':>14} {'% of image':>12} {'Description'}")
print(f"  {'─'*65}")
for t_low, t_high, desc in canny_configs:
    edges    = cv2.Canny(blurred, t_low, t_high)
    n_edge   = cv2.countNonZero(edges)
    pct      = 100 * n_edge / edges.size
    print(f"  {t_low:>6} {t_high:>7} {n_edge:>14,} {pct:>12.2f}% {desc}")
print()
print(f"  Canny pipeline steps (for threshold1=50, threshold2=150):")
edges_standard = cv2.Canny(blurred, 50, 150)
print(f"  1. Gaussian blur (already applied)")
print(f"  2. Sobel: Gmax={Gmag.max():.0f}, mean strong gradient={Gmag[Gmag>50].mean():.0f}")
print(f"  3. Non-maximum suppression: thins edges to 1 pixel wide")
print(f"  4. Double threshold: T_low=50, T_high=150")
print(f"  5. Hysteresis: connects weak edges to strong ones")
print(f"  Result: {cv2.countNonZero(edges_standard):,} edge pixels "
      f"({100*cv2.countNonZero(edges_standard)/edges_standard.size:.2f}% of image)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Thresholding methods
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Thresholding: Simple, Otsu, Adaptive")
print("━" * 65)
print()

# Simple threshold at multiple values
print(f"  Simple thresholding at different values:")
for t in [64, 100, 128, 160, 200]:
    _, binary = cv2.threshold(clean, t, 255, cv2.THRESH_BINARY)
    fg_pct    = 100 * cv2.countNonZero(binary) / binary.size
    print(f"    threshold={t:>3}: {cv2.countNonZero(binary):>7,} fg pixels  ({fg_pct:.1f}%)")
print()

# Otsu automatic threshold
otsu_thresh, binary_otsu = cv2.threshold(clean, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
fg_otsu = 100 * cv2.countNonZero(binary_otsu) / binary_otsu.size
print(f"  Otsu's method: auto-selected threshold = {otsu_thresh:.0f}")
print(f"    Foreground pixels: {cv2.countNonZero(binary_otsu):,}  ({fg_otsu:.1f}%)")
print(f"    Minimises between-class variance of two pixel classes")
print()

# Adaptive threshold — simulate uneven illumination
uneven = clean.copy().astype(np.float32)
# Add a gradient illumination (brighter on left)
grad_illum = np.tile(np.linspace(1.5, 0.5, clean.shape[1]), (clean.shape[0], 1))
uneven     = np.clip(uneven * grad_illum, 0, 255).astype(np.uint8)

_, binary_otsu_uneven  = cv2.threshold(uneven, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_adaptive = cv2.adaptiveThreshold(uneven, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

# Count correctly classified pixels (compare to clean binary)
_, clean_binary = cv2.threshold(clean, 128, 255, cv2.THRESH_BINARY)
acc_otsu = 100 * (binary_otsu_uneven == clean_binary).mean()
acc_adap = 100 * (binary_adaptive    == clean_binary).mean()
print(f"  Uneven illumination comparison:")
print(f"    Otsu on uneven image:    accuracy vs clean binary = {acc_otsu:.1f}%")
print(f"    Adaptive (11×11 blocks): accuracy vs clean binary = {acc_adap:.1f}%")
print(f"    Adaptive wins when illumination varies across the image")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Morphological operations
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Morphological operations on binary images")
print("━" * 65)
print()

# Create a binary image with noise (salt-and-pepper in binary context)
_, binary_clean = cv2.threshold(clean, 128, 255, cv2.THRESH_BINARY)
# Add isolated noise pixels
noisy_binary = binary_clean.copy()
noise_pixels  = rng.integers(0, binary_clean.size, size=500)
noisy_binary.flat[noise_pixels] = 255 - noisy_binary.flat[noise_pixels]
# Add holes in objects
for _ in range(10):
    x, y = rng.integers(30, 200, size=2)
    noisy_binary[y:y+4, x:x+4] = 0

rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
rect5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ell5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

ops = {
    "Original":      noisy_binary,
    "Erode 3×3":     cv2.erode(noisy_binary, rect3, iterations=1),
    "Dilate 3×3":    cv2.dilate(noisy_binary, rect3, iterations=1),
    "Open 3×3":      cv2.morphologyEx(noisy_binary, cv2.MORPH_OPEN, rect3),
    "Close 3×3":     cv2.morphologyEx(noisy_binary, cv2.MORPH_CLOSE, rect3),
    "Open+Close":    cv2.morphologyEx(
                         cv2.morphologyEx(noisy_binary, cv2.MORPH_OPEN, rect3),
                         cv2.MORPH_CLOSE, rect5),
    "Gradient 3×3":  cv2.morphologyEx(noisy_binary, cv2.MORPH_GRADIENT, rect3),
}

orig_fg = cv2.countNonZero(binary_clean)
print(f"  Binary image morphology (target foreground: {orig_fg:,} px):")
print(f"  {'Operation':<14} {'FG pixels':>12} {'Δ vs target':>14} {'Effect'}")
print(f"  {'─'*65}")
effects = {
    "Original":   "with noise/holes",
    "Erode 3×3":  "shrinks objects, removes tiny specks",
    "Dilate 3×3": "grows objects, fills small holes",
    "Open 3×3":   "removes noise (erode→dilate)",
    "Close 3×3":  "fills holes (dilate→erode)",
    "Open+Close":  "remove noise AND fill holes",
    "Gradient 3×3": "boundary/outline of objects",
}
for name, result in ops.items():
    fg  = cv2.countNonZero(result)
    delta = fg - orig_fg
    print(f"  {name:<14} {fg:>12,} {delta:>+14,} {effects.get(name,'')}")
print()

# Connected components on cleaned binary
cleaned   = cv2.morphologyEx(noisy_binary, cv2.MORPH_OPEN, rect3)
cleaned   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, rect5)
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

print(f"  Connected components after morphological cleaning:")
print(f"    Total components (including background): {n_labels}")
print(f"    Foreground components: {n_labels - 1}")
if n_labels > 1:
    areas = stats[1:, cv2.CC_STAT_AREA]   # skip background (label 0)
    print(f"    Area range:  [{areas.min()}, {areas.max()}]")
    print(f"    Mean area:   {areas.mean():.0f} pixels")
    large = (areas > 100).sum()
    print(f"    Large (>100px): {large} components")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Feature Detection — Corners, SIFT, ORB, and Matching": {
        "description": (
            "Classical feature detection and description pipeline. "
            "Harris corner detector: response map and threshold. "
            "Shi-Tomasi: goodFeaturesToTrack for tracking applications. "
            "SIFT keypoints: scale-invariant detection and 128-dim descriptors. "
            "ORB keypoints: FAST + BRIEF binary descriptors. "
            "Brute-force feature matching with Hamming and L2 distance. "
            "Lowe's ratio test for removing ambiguous matches. "
            "RANSAC homography estimation from matched features. "
            "Geometric transforms: rotation, scaling, perspective warp. "
            "Feature matching performance benchmark across detectors. "
            "Repeatability analysis under image transformations."
        ),
        "language": "python",
        "code": '''
import numpy as np
import cv2
import time

print("=" * 65)
print("  FEATURE DETECTION — CORNERS, SIFT, ORB, AND MATCHING")
print("=" * 65)
print()

rng = np.random.default_rng(42)

# ── Create synthetic scene with identifiable features ─────────────────
def make_feature_rich_image(h=320, w=320, seed=42):
    np.random.seed(seed)
    img = np.ones((h, w, 3), dtype=np.uint8) * 200  # light grey background

    # Draw various geometric shapes (rich in corners/edges)
    cv2.rectangle(img, (20,  20),  (90,  90),  (50,  50, 180), -1)
    cv2.rectangle(img, (200, 20),  (280, 80),  (50, 170,  50), -1)
    cv2.circle(img,    (160, 200), 50,          (180, 50,  50), -1)
    cv2.ellipse(img,   (80, 220),  (40, 20), 30, 0, 360, (100, 180, 100), -1)

    # Add checkerboard pattern (rich in corners)
    for row in range(0, 60, 20):
        for col in range(220, 300, 20):
            if (row // 20 + col // 20) % 2 == 0:
                cv2.rectangle(img, (col, row+10), (col+18, row+28), (30, 30, 30), -1)

    # Random textures (helps feature distinctiveness)
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def apply_transform(img, angle=15, scale=0.9, tx=20, ty=15):
    """Apply a known affine transform for repeatability testing."""
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=200), M

img1        = make_feature_rich_image()
img2, M_gt  = apply_transform(img1, angle=12, scale=0.88, tx=15, ty=10)
grey1       = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grey2       = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print(f"  Scene image: {img1.shape}")
print(f"  Transform applied: rotation=12°, scale=0.88, translation=(15,10)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Harris corner detector
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Harris corner detector: response map analysis")
print("━" * 65)
print()

t0  = time.perf_counter()
dst = cv2.cornerHarris(grey1, blockSize=2, ksize=3, k=0.04)
t_harris = (time.perf_counter() - t0) * 1000

# Normalise and threshold
dst_norm = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply non-maximum suppression manually
def nms_corners(response, threshold_pct=0.01, min_dist=10):
    thresh    = threshold_pct * response.max()
    ys, xs    = np.where(response > thresh)
    scores    = response[ys, xs]
    order     = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    keep      = []
    for i in range(len(ys)):
        if all(abs(ys[i]-ys[j]) > min_dist or abs(xs[i]-xs[j]) > min_dist
               for j in keep):
            keep.append(i)
    return list(zip(xs[keep], ys[keep], scores[keep]))

corners = nms_corners(dst, threshold_pct=0.01, min_dist=8)
print(f"  Harris corner detector (blockSize=2, ksize=3, k=0.04):")
print(f"    Response map: min={dst.min():.4f}, max={dst.max():.4f}")
print(f"    Detected corners (after NMS): {len(corners)}")
print(f"    Compute time: {t_harris:.2f}ms")
print()
print(f"  Harris response interpretation:")
print(f"    R >> 0  (corner):  {(dst > 0.01*dst.max()).sum():,} pixels")
print(f"    R << 0  (edge):    {(dst < -0.001*abs(dst).max()).sum():,} pixels")
print(f"    R ≈ 0   (flat):    {(np.abs(dst) <= 0.001*abs(dst).max()).sum():,} pixels")
print()

# Top corners
print(f"  Top 10 corner locations (x, y, score):")
for i, (x, y, s) in enumerate(sorted(corners, key=lambda c: -c[2])[:10]):
    print(f"    [{i+1}] x={x:>4}, y={y:>4}, score={s:.4f}")
print()

# Shi-Tomasi (goodFeaturesToTrack)
t0    = time.perf_counter()
pts_st = cv2.goodFeaturesToTrack(grey1, maxCorners=100, qualityLevel=0.01,
                                   minDistance=10, useHarrisDetector=False)
t_st = (time.perf_counter() - t0) * 1000
n_st = len(pts_st) if pts_st is not None else 0
print(f"  Shi-Tomasi (goodFeaturesToTrack):")
print(f"    Detected: {n_st} corners in {t_st:.2f}ms")
print(f"    Criterion: minimise the MINIMUM eigenvalue (more stable than Harris R)")
print(f"    Use for: Lucas-Kanade optical flow tracking")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: SIFT keypoints and descriptors
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — SIFT: scale-invariant keypoints and descriptors")
print("━" * 65)
print()

sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04,
                        edgeThreshold=10, sigma=1.6)

t0   = time.perf_counter()
kp1_sift, desc1_sift = sift.detectAndCompute(grey1, None)
t_sift = (time.perf_counter() - t0) * 1000

t0   = time.perf_counter()
kp2_sift, desc2_sift = sift.detectAndCompute(grey2, None)
_ = time.perf_counter() - t0

print(f"  SIFT on image 1:")
print(f"    Keypoints detected: {len(kp1_sift)}")
print(f"    Descriptor shape:   {desc1_sift.shape}  (n_kp × 128)")
print(f"    Detect+compute time: {t_sift:.1f}ms")
print()

# Analyse keypoint properties
sizes    = [kp.size for kp in kp1_sift]
angles   = [kp.angle for kp in kp1_sift]
responses = [kp.response for kp in kp1_sift]
print(f"  SIFT keypoint statistics:")
print(f"    Scale (kp.size): min={min(sizes):.1f}, max={max(sizes):.1f}, "
      f"mean={np.mean(sizes):.1f}")
print(f"    Angle (degrees): {np.mean(angles):.1f}° mean  "
      f"(full 0-360° = rotation invariant)")
print(f"    Response:        min={min(responses):.4f}, max={max(responses):.4f}")
print()

# Descriptor statistics
print(f"  SIFT descriptor statistics:")
print(f"    dim:  128 (4×4 grid × 8 orientation bins)")
print(f"    norm: {np.linalg.norm(desc1_sift, axis=1).mean():.3f} "
      f"±{np.linalg.norm(desc1_sift, axis=1).std():.3f} (L2-normalised to 512)")
print(f"    range: [{desc1_sift.min():.0f}, {desc1_sift.max():.0f}]")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: ORB keypoints and binary descriptors
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — ORB: fast binary descriptor with FAST+BRIEF")
print("━" * 65)
print()

orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8,
                      edgeThreshold=31, firstLevel=0, WTA_K=2,
                      scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)

t0   = time.perf_counter()
kp1_orb, desc1_orb = orb.detectAndCompute(grey1, None)
t_orb = (time.perf_counter() - t0) * 1000

kp2_orb, desc2_orb = orb.detectAndCompute(grey2, None)

print(f"  ORB on image 1:")
print(f"    Keypoints detected: {len(kp1_orb)}")
print(f"    Descriptor shape:   {desc1_orb.shape}  (n_kp × 32 bytes = 256 bits)")
print(f"    Detect+compute time: {t_orb:.1f}ms  (vs SIFT: {t_sift:.1f}ms)")
print()
print(f"  ORB vs SIFT comparison:")
print(f"  {'Property':<25} {'SIFT':>15} {'ORB':>15}")
print(f"  {'─'*58}")
print(f"  {'Descriptor type':<25} {'float32':>15} {'uint8 (binary)':>15}")
print(f"  {'Descriptor dim':<25} {'128 floats':>15} {'256 bits (32B)':>15}")
print(f"  {'Memory per kp':<25} {'512 bytes':>15} {'32 bytes':>15}")
print(f"  {'Match distance':<25} {'L2 (euclidean)':>15} {'Hamming (XOR)':>15}")
print(f"  {'Speed':<25} {'slower':>15} {'10-100× faster':>15}")
print(f"  {'Scale invariant':<25} {'Yes':>15} {'Yes (pyramid)':>15}")
print(f"  {'Rotation invariant':<25} {'Yes':>15} {'Yes (IC angle)':>15}")
print(f"  {'Patent':<25} {'Free (2020)':>15} {'Always free':>15}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Feature matching and Lowe's ratio test
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Feature matching: BF matcher and ratio test")
print("━" * 65)
print()

# ORB matching (Hamming distance)
bf_orb   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
t0       = time.perf_counter()
matches_orb_all = bf_orb.knnMatch(desc1_orb, desc2_orb, k=2)
t_match_orb = (time.perf_counter() - t0) * 1000

# Apply Lowe's ratio test
ratio_thresh   = 0.75
good_orb_75    = [m for m, n in matches_orb_all if m.distance < ratio_thresh * n.distance]
good_orb_85    = [m for m, n in matches_orb_all if m.distance < 0.85 * n.distance]

print(f"  ORB matching ({len(kp1_orb)} kp1 × {len(kp2_orb)} kp2):")
print(f"    All knn matches:          {len(matches_orb_all)}")
print(f"    After ratio test (0.75):  {len(good_orb_75)}  "
      f"({100*len(good_orb_75)/len(matches_orb_all):.1f}% kept)")
print(f"    After ratio test (0.85):  {len(good_orb_85)}  "
      f"({100*len(good_orb_85)/len(matches_orb_all):.1f}% kept)")
print(f"    Match time: {t_match_orb:.2f}ms")
print()

# Distance statistics on good matches
if good_orb_75:
    dists = [m.distance for m in good_orb_75]
    print(f"  Match distance statistics (Hamming, 0=perfect, 256=max):")
    print(f"    min={min(dists):.0f}, max={max(dists):.0f}, "
          f"mean={np.mean(dists):.1f}, median={np.median(dists):.1f}")
    dist_thresholds = [32, 48, 64, 80, 96]
    print(f"  Matches by distance threshold:")
    for dt in dist_thresholds:
        n = sum(1 for m in good_orb_75 if m.distance <= dt)
        print(f"    ≤{dt}: {n:>4}  ({100*n/len(good_orb_75):.1f}%)")
print()

# SIFT matching (L2 distance)
bf_sift  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches_sift_all = bf_sift.knnMatch(desc1_sift, desc2_sift, k=2)
good_sift = [m for m, n in matches_sift_all if m.distance < 0.75 * n.distance]

print(f"  SIFT matching ({len(kp1_sift)} kp1 × {len(kp2_sift)} kp2):")
print(f"    Good matches (ratio test 0.75): {len(good_sift)}")
if good_sift:
    sift_dists = [m.distance for m in good_sift]
    print(f"    Distance: mean={np.mean(sift_dists):.1f}, "
          f"std={np.std(sift_dists):.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Homography estimation with RANSAC
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Homography with RANSAC: geometric verification")
print("━" * 65)
print()

if len(good_sift) >= 4:
    src_pts  = np.float32([kp1_sift[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
    dst_pts  = np.float32([kp2_sift[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

    t0       = time.perf_counter()
    H, mask  = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                   ransacReprojThreshold=5.0)
    t_ransac = (time.perf_counter() - t0) * 1000

    inliers  = mask.ravel().sum()
    outliers = mask.size - inliers

    print(f"  RANSAC homography estimation:")
    print(f"    Input matches:  {len(good_sift)}")
    print(f"    Inliers:        {inliers}  ({100*inliers/len(good_sift):.1f}%)")
    print(f"    Outliers:       {outliers}  (rejected as mismatches)")
    print(f"    Compute time:   {t_ransac:.2f}ms")
    print()
    print(f"  Estimated homography matrix H:")
    for row in H:
        print(f"    {row}")
    print()

    # Evaluate homography accuracy by applying it to known corners
    h1, w1  = grey1.shape
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    corners2 = cv2.perspectiveTransform(corners1, H)
    corners2 = corners2.reshape(4, 2)

    # Apply ground truth transform for comparison
    corners2_gt = []
    for pt in corners1.reshape(4, 2):
        pt_h = np.array([pt[0], pt[1], 1.0])
        M_3x3 = np.vstack([M_gt, [0,0,1]])
        pt_t  = M_3x3 @ pt_h
        corners2_gt.append(pt_t[:2])
    corners2_gt = np.array(corners2_gt)

    errors = np.linalg.norm(corners2 - corners2_gt, axis=1)
    print(f"  Corner reprojection errors (H vs ground truth):")
    for i, (err, corner) in enumerate(zip(errors, corners1.reshape(4,2))):
        print(f"    Corner {i} ({corner[0]:.0f},{corner[1]:.0f}): "
              f"error = {err:.2f} pixels")
    print(f"  Mean reprojection error: {errors.mean():.2f} pixels")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Feature detector performance summary
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Feature detector benchmark and selection guide")
print("━" * 65)
print()

detectors = {}
for name, detector in [
    ("SIFT",   cv2.SIFT_create(nfeatures=500)),
    ("ORB",    cv2.ORB_create(nfeatures=500)),
    ("AKAZE",  cv2.AKAZE_create()),
]:
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        kp, desc = detector.detectAndCompute(grey1, None)
        times.append((time.perf_counter() - t0) * 1000)
    detectors[name] = {
        "n_kp":     len(kp),
        "desc_dim": desc.shape[1] if desc is not None else 0,
        "dtype":    str(desc.dtype) if desc is not None else "N/A",
        "time_ms":  np.mean(times),
    }

print(f"  {'Detector':<8} {'Keypoints':>12} {'Desc dim':>10} "
      f"{'Dtype':<12} {'Time (ms)':>12}")
print(f"  {'─'*58}")
for name, info in detectors.items():
    print(f"  {name:<8} {info['n_kp']:>12} {info['desc_dim']:>10} "
          f"{info['dtype']:<12} {info['time_ms']:>12.2f}")

print()
print(f"  Selection guide:")
print(f"  ┌──────────────────────────────────────────────────────────────────┐")
print(f"  │ Use case                     │ Detector │ Why                    │")
print(f"  ├──────────────────────────────────────────────────────────────────┤")
print(f"  │ Real-time tracking           │ ORB      │ Fastest, binary match  │")
print(f"  │ Panorama stitching           │ SIFT     │ Scale-robust, accurate │")
print(f"  │ Object recognition           │ SIFT     │ Distinctive 128-dim    │")
print(f"  │ Embedded / resource-limited  │ ORB/AKAZE│ No patent, fast        │")
print(f"  │ Document scanning            │ SIFT     │ Handles perspective    │")
print(f"  │ Optical flow (LK)            │ Shi-Tom  │ Uniform distribution   │")
print(f"  └──────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Optical Flow, Contours & Transforms — Motion and Geometry": {
        "description": (
            "Motion analysis, shape description, and geometric transforms. "
            "Lucas-Kanade sparse optical flow: point tracking simulation. "
            "Farneback dense optical flow: HSV flow visualisation. "
            "Background subtraction with MOG2 on synthetic video. "
            "Contour detection: findContours hierarchy and modes. "
            "Shape descriptors: area, perimeter, circularity, solidity. "
            "Polygon approximation for shape classification (triangle/quad). "
            "Minimum bounding rectangle and enclosing circle. "
            "Affine transform: rotation, scale, shear. "
            "Perspective transform: document dewarping simulation. "
            "Template matching: TM_CCOEFF_NORMED with multi-scale search. "
            "HOG descriptor computation and visualisation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import cv2
import time

print("=" * 65)
print("  OPTICAL FLOW, CONTOURS & TRANSFORMS — MOTION AND GEOMETRY")
print("=" * 65)
print()

rng = np.random.default_rng(0)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Lucas-Kanade optical flow (sparse tracking)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Lucas-Kanade sparse optical flow")
print("━" * 65)
print()

def make_frame_with_objects(h=200, w=200, offset_x=0, offset_y=0, t=0):
    """Synthetic frame with moving objects."""
    frame = np.zeros((h, w, 3), dtype=np.uint8) + 40
    # Moving rectangle
    x1 = 30 + offset_x
    y1 = 30 + offset_y
    cv2.rectangle(frame, (x1, y1), (x1+50, y1+40), (100, 180, 100), -1)
    # Rotating circle
    cx = 140 + int(20 * np.cos(t))
    cy = 100 + int(20 * np.sin(t))
    cv2.circle(frame, (cx, cy), 20, (180, 80, 80), -1)
    # Static background texture
    for i in range(0, w, 30):
        cv2.line(frame, (i, 0), (i, h), (70, 70, 70), 1)
    return frame

# Simulate 5 frames of motion
N_FRAMES    = 10
TRUE_DX, TRUE_DY = 3, 2   # true motion per frame

frames = [make_frame_with_objects(t=i*0.3, offset_x=i*TRUE_DX,
                                   offset_y=i*TRUE_DY)
          for i in range(N_FRAMES)]
greys  = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

# Detect points to track in first frame
pts0 = cv2.goodFeaturesToTrack(greys[0], maxCorners=50, qualityLevel=0.01,
                                minDistance=10, blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

print(f"  Initial tracked points: {len(pts0)}")
print(f"  True motion per frame: dx={TRUE_DX}, dy={TRUE_DY}")
print()
print(f"  Lucas-Kanade tracking trace:")
print(f"  {'Frame':>6} {'Found pts':>12} {'Mean dx':>10} {'Mean dy':>10} {'Error':>10}")
print(f"  {'─'*52}")

current_pts = pts0.copy()
total_tracked = 0
for frame_idx in range(1, N_FRAMES):
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        greys[frame_idx-1], greys[frame_idx], current_pts, None, **lk_params
    )
    # Keep only successfully tracked points
    good_new = next_pts[status.ravel() == 1]
    good_old = current_pts[status.ravel() == 1]

    if len(good_old) > 0:
        flow_vecs = good_new - good_old
        mean_dx   = flow_vecs[:, 0, 0].mean()
        mean_dy   = flow_vecs[:, 0, 1].mean()
        error     = np.sqrt((mean_dx - TRUE_DX)**2 + (mean_dy - TRUE_DY)**2)
        print(f"  {frame_idx:>6} {len(good_new):>12} {mean_dx:>10.2f} "
              f"{mean_dy:>10.2f} {error:>10.3f}")
        total_tracked += len(good_new)
        current_pts = good_new.reshape(-1, 1, 2)

print(f"\n  Average tracked points over {N_FRAMES-1} frames: "
      f"{total_tracked/(N_FRAMES-1):.0f}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Farneback dense optical flow
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Farneback dense optical flow")
print("━" * 65)
print()

prev_grey = greys[0]
next_grey = greys[3]   # 3 frames apart = dx=9, dy=6

t0  = time.perf_counter()
flow = cv2.calcOpticalFlowFarneback(
    prev_grey, next_grey, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
t_farneback = (time.perf_counter() - t0) * 1000

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

print(f"  Farneback dense flow ({prev_grey.shape} image):")
print(f"    Flow array shape: {flow.shape}  (H, W, 2) — (u,v) per pixel")
print(f"    Compute time: {t_farneback:.1f}ms")
print()
print(f"  Flow statistics:")
print(f"    u (horizontal): mean={flow[...,0].mean():.2f}, "
      f"std={flow[...,0].std():.2f}")
print(f"    v (vertical):   mean={flow[...,1].mean():.2f}, "
      f"std={flow[...,1].std():.2f}")
print(f"    magnitude:      mean={mag.mean():.2f}, max={mag.max():.2f}")
print(f"    expected:       mean ≈ {np.sqrt(TRUE_DX**2 + TRUE_DY**2)*3:.2f}")
print()

# Flow direction histogram
ang_valid = ang[mag > 1.0]
bins_flow, _ = np.histogram(ang_valid, bins=8, range=(0, 360))
print(f"  Flow direction histogram (where motion > 1px):")
dir_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
for lbl, cnt in zip(dir_labels, bins_flow):
    bar = "█" * (cnt // max(bins_flow.max()//15, 1))
    print(f"    {lbl:>3}: {cnt:>6}  {bar}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Contour analysis and shape descriptors
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Contour analysis: shape descriptors")
print("━" * 65)
print()

def make_shapes_image(h=300, w=400):
    """Create image with known geometric shapes for contour analysis."""
    img = np.zeros((h, w), dtype=np.uint8)
    # Perfect circle
    cv2.circle(img, (60, 60), 40, 255, -1)
    # Rectangle
    cv2.rectangle(img, (130, 30), (230, 100), 255, -1)
    # Triangle (polygon)
    tri = np.array([[280, 20], [360, 20], [320, 100]], dtype=np.int32)
    cv2.fillPoly(img, [tri], 255)
    # Ellipse
    cv2.ellipse(img, (70, 180), (50, 25), 20, 0, 360, 255, -1)
    # Pentagon (approx circle)
    n_pts = 5
    pts_pent = np.array([
        [int(150 + 40*np.cos(2*np.pi*i/n_pts - np.pi/2)),
         int(180 + 40*np.sin(2*np.pi*i/n_pts - np.pi/2))]
        for i in range(n_pts)], dtype=np.int32)
    cv2.fillPoly(img, [pts_pent], 255)
    # Star (complex shape)
    star_pts = []
    for i in range(10):
        r = 40 if i % 2 == 0 else 20
        angle = i * np.pi / 5 - np.pi / 2
        star_pts.append([int(310 + r*np.cos(angle)), int(180 + r*np.sin(angle))])
    cv2.fillPoly(img, [np.array(star_pts, np.int32)], 255)
    return img

shapes = make_shapes_image()
contours, hierarchy = cv2.findContours(shapes, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

print(f"  Found {len(contours)} contours")
print()
print(f"  {'Shape':<12} {'Area':>8} {'Perim':>8} {'Circ':>7} "
      f"{'Solid':>7} {'Aspect':>8} {'Vertices':>10}")
print(f"  {'─'*65}")

shape_names = ["Circle", "Rectangle", "Triangle", "Ellipse", "Pentagon", "Star"]

for i, cnt in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
    area = cv2.contourArea(cnt)
    if area < 100: continue
    perim = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    cx = int(M['m10'] / (M['m00'] + 1e-8))
    cy = int(M['m01'] / (M['m00'] + 1e-8))

    # Shape descriptors
    circularity = 4 * np.pi * area / (perim**2 + 1e-8)
    hull_area   = cv2.contourArea(cv2.convexHull(cnt))
    solidity    = area / (hull_area + 1e-8)
    x,y,w,h    = cv2.boundingRect(cnt)
    aspect      = w / (h + 1e-8)

    # Polygon approximation
    epsilon = 0.03 * perim
    approx  = cv2.approxPolyDP(cnt, epsilon, True)
    n_verts = len(approx)

    shape_name = shape_names[i] if i < len(shape_names) else f"Shape_{i}"
    print(f"  {shape_name:<12} {area:>8.0f} {perim:>8.1f} {circularity:>7.3f} "
          f"{solidity:>7.3f} {aspect:>8.2f} {n_verts:>10}")

print()
print(f"  Shape classification rules:")
print(f"    Circularity ≈ 1.0:           circle / ellipse")
print(f"    Solidity ≈ 1.0, aspect ≈ 1:  convex blob")
print(f"    Solidity < 0.8:              star / concave shape")
print(f"    Vertices == 3:               triangle")
print(f"    Vertices == 4:               rectangle / quadrilateral")
print(f"    Vertices == 5:               pentagon")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Geometric transforms
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Geometric transforms: affine and perspective")
print("━" * 65)
print()

base = np.zeros((200, 300, 3), dtype=np.uint8) + 50
cv2.rectangle(base, (50, 50), (150, 150), (100, 180, 100), -1)
cv2.circle(base, (230, 100), 40, (180, 80, 80), -1)
cv2.putText(base, "TEST", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 2)

h, w = base.shape[:2]

# Rotation
angle = 25
M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
rotated = cv2.warpAffine(base, M_rot, (w, h))
print(f"  Rotation ({angle}°, centre=({w//2},{h//2})):")
print(f"    M = {M_rot.round(4)}")
print()

# Scale + rotation
M_scale = cv2.getRotationMatrix2D((w//2, h//2), 10, 0.75)
scaled  = cv2.warpAffine(base, M_scale, (w, h))
print(f"  Scale (0.75×) + rotation (10°):")
print(f"    M = {M_scale.round(4)}")
print()

# General affine from 3 points
src_tri = np.float32([[0,0], [w-1,0], [0,h-1]])
dst_tri = np.float32([[w*0.1, h*0.1], [w*0.85, h*0.05], [w*0.15, h*0.85]])
M_aff   = cv2.getAffineTransform(src_tri, dst_tri)
affine  = cv2.warpAffine(base, M_aff, (w, h))
print(f"  Affine from 3 points (includes shear):")
print(f"    Parallel lines preserved, angles may change")
print()

# Perspective (4-point transform) — document dewarping simulation
skewed = np.zeros_like(base)
src_quad = np.float32([[30, 20], [w-40, 10], [w-20, h-30], [20, h-20]])
dst_quad = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
cv2.fillPoly(skewed, [src_quad.astype(np.int32)], (200, 200, 200))
cv2.putText(skewed, "DOC", (int(w*0.3), int(h*0.5)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 200), 3)

M_persp   = cv2.getPerspectiveTransform(src_quad, dst_quad)
dewarped  = cv2.warpPerspective(skewed, M_persp, (w, h))
print(f"  Perspective transform (document dewarping):")
print(f"    Source quad: {src_quad.tolist()}")
print(f"    Dest quad: axis-aligned rectangle")
print(f"    H matrix (3×3):")
for row in M_persp:
    print(f"      {row.round(4)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Template matching
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Template matching and multi-scale search")
print("━" * 65)
print()

# Create scene and template
scene = np.zeros((200, 300), dtype=np.uint8) + 128
target_centre = (180, 100)
cv2.circle(scene, target_centre, 20, 200, -1)
cv2.rectangle(scene, (50, 60), (100, 130), 80, -1)
# Add noise
scene_noisy = cv2.add(scene,
    np.clip(rng.integers(-20, 20, scene.shape), -255, 255).astype(np.int8).view(np.uint8))

# Extract template from scene
ty, tx = target_centre[1] - 25, target_centre[0] - 25
template = scene[ty:ty+50, tx:tx+50].copy()

# Template matching
t0     = time.perf_counter()
result = cv2.matchTemplate(scene_noisy, template, cv2.TM_CCOEFF_NORMED)
t_tm   = (time.perf_counter() - t0) * 1000

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
found_x  = max_loc[0] + template.shape[1]//2
found_y  = max_loc[1] + template.shape[0]//2
error_x  = abs(found_x - target_centre[0])
error_y  = abs(found_y - target_centre[1])

print(f"  Template: {template.shape}, Method: TM_CCOEFF_NORMED")
print(f"  Result map shape: {result.shape}")
print(f"  Best match score: {max_val:.4f}  (1.0 = perfect)")
print(f"  Found location:   ({found_x}, {found_y})")
print(f"  True location:    {target_centre}")
print(f"  Error:            ({error_x}, {error_y}) pixels")
print(f"  Matching time:    {t_tm:.2f}ms")
print()

# Multi-scale template matching
scales  = [0.75, 0.90, 1.00, 1.10, 1.25]
results = []
for sc in scales:
    tw = max(1, int(template.shape[1] * sc))
    th = max(1, int(template.shape[0] * sc))
    if tw < 5 or th < 5:
        continue
    scaled_tmpl = cv2.resize(template, (tw, th))
    if scaled_tmpl.shape[0] > scene_noisy.shape[0] or \
       scaled_tmpl.shape[1] > scene_noisy.shape[1]:
        continue
    res = cv2.matchTemplate(scene_noisy, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
    _, best_score, _, best_loc = cv2.minMaxLoc(res)
    fx = best_loc[0] + tw//2
    fy = best_loc[1] + th//2
    results.append((sc, best_score, fx, fy))

print(f"  Multi-scale template matching:")
print(f"  {'Scale':>7} {'Score':>8} {'Found x':>10} {'Found y':>10} {'Error':>8}")
print(f"  {'─'*46}")
for sc, score, fx, fy in results:
    err = np.sqrt((fx-target_centre[0])**2 + (fy-target_centre[1])**2)
    marker = " ← best" if sc == 1.0 else ""
    print(f"  {sc:>7.2f} {score:>8.4f} {fx:>10} {fy:>10} {err:>8.2f}px{marker}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: HOG descriptor
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — HOG descriptor: Histogram of Oriented Gradients")
print("━" * 65)
print()

# HOG for a 64×128 window (standard pedestrian detection size)
WIN_SIZE  = (64, 128)
CELL_SIZE = (8, 8)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
N_BINS = 9

hog = cv2.HOGDescriptor(
    WIN_SIZE,   BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, N_BINS
)

# Compute HOG on a test patch
test_patch = np.zeros(WIN_SIZE[::-1], dtype=np.uint8)  # 128×64
cv2.rectangle(test_patch, (15, 30), (49, 100), 200, -1)
cv2.line(test_patch, (32, 10), (32, 118), 255, 4)

t0          = time.perf_counter()
descriptor  = hog.compute(test_patch)
t_hog       = (time.perf_counter() - t0) * 1000

# Compute expected descriptor length
n_cells_x    = WIN_SIZE[0] // CELL_SIZE[0]
n_cells_y    = WIN_SIZE[1] // CELL_SIZE[1]
n_blocks_x   = (n_cells_x - 1)   # with stride=1 cell
n_blocks_y   = (n_cells_y - 1)
n_blocks_x2  = (WIN_SIZE[0] - BLOCK_SIZE[0]) // BLOCK_STRIDE[0] + 1
n_blocks_y2  = (WIN_SIZE[1] - BLOCK_SIZE[1]) // BLOCK_STRIDE[1] + 1
expected_len = n_blocks_x2 * n_blocks_y2 * 4 * N_BINS

print(f"  HOG descriptor configuration:")
print(f"    Window size:    {WIN_SIZE}  (W × H)")
print(f"    Cell size:      {CELL_SIZE}  → {WIN_SIZE[0]//CELL_SIZE[0]} × "
      f"{WIN_SIZE[1]//CELL_SIZE[1]} cells = "
      f"{(WIN_SIZE[0]//CELL_SIZE[0])*(WIN_SIZE[1]//CELL_SIZE[1])} cells")
print(f"    Block size:     {BLOCK_SIZE}  = 2×2 cells")
print(f"    Block stride:   {BLOCK_STRIDE}  = 1 cell overlap")
print(f"    Bins per cell:  {N_BINS}  (0°–180° in 20° steps)")
print(f"    Descriptor len: {len(descriptor)}  "
      f"(= {n_blocks_x2}×{n_blocks_y2} blocks × 4 cells × {N_BINS} bins)")
print(f"    Compute time:   {t_hog:.2f}ms")
print()
print(f"  Descriptor statistics:")
print(f"    mean  = {descriptor.mean():.4f}")
print(f"    std   = {descriptor.std():.4f}")
print(f"    max   = {descriptor.max():.4f}  (L2-block-normalised)")
print(f"    % nonzero = {100*(descriptor > 0.001).mean():.1f}%")
print()

# Pedestrian detector (pre-trained HOG+SVM)
hog_people = cv2.HOGDescriptor()
hog_people.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print(f"  Pre-trained pedestrian detector loaded:")
print(f"    SVM descriptor: {len(hog_people.getSVMDetector())} dimensions")
print(f"    Usage: rects, weights = hog.detectMultiScale(image, winStride=(8,8),")
print(f"                                                   padding=(4,4), scale=1.05)")
print(f"    Outputs: list of (x,y,w,h) bounding boxes + confidence scores")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }