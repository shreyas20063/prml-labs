# Lab 4 Study Notes (Viva Prep): Multiclass Logistic Regression and SVMs

These notes explain every block of `Lab4-EC24BT033.ipynb`: first the problem, then the approach, then what each code block does and the theory behind it. The goal is that you can defend every decision in the viva without memorizing syntax.

---

## The big picture

Lab 4 has two halves, and both are about **linear classifiers**, meaning models whose decision boundary is a straight line (in 2D) or a flat plane (in higher dimensions):

1. **Question 1**: extend logistic regression from 2 classes to 3+ classes using **softmax**, built completely from scratch.
2. **Question 2**: study **Support Vector Machines (SVMs)**, which pick the boundary in a completely different way: not by probabilities, but by geometry (maximizing the margin).

The deep connection to earlier labs: both models have a **regularization knob** (sklearn's C), and Lab 4 keeps showing the same lesson as Labs 2 and 3: a moderately regularized model generalizes best.

---

# Question 1: Multiclass Logistic Regression via Softmax

## The problem

The Wine dataset: 178 samples, 13 numeric features (chemical measurements like alcohol, flavanoids), and **3 classes** (three grape cultivars). We must predict which cultivar a wine belongs to.

Why is this not just Lab 3 again? In Lab 3, logistic regression used the **sigmoid**, which outputs ONE probability: P(class 1). The other class gets 1 minus that. With 3 classes that trick breaks: we need three probabilities that are all non-negative and **sum to 1**. That is exactly what softmax provides.

## The approach

1. Load and inspect the data (class balance matters for splitting and metric choice).
2. Split 70:15:15 with **stratification** (from scratch, seed 42).
3. Standardize using **training statistics only** (no leakage).
4. Build softmax, cross-entropy cost, and its gradient from scratch; train with batch gradient descent at three learning rates; pick the best by **validation cross-entropy**.
5. Prove numerically that softmax with K = 2 collapses to the sigmoid (this shows softmax is the true generalization, not a different model).
6. Evaluate on the test set with manually computed metrics and compare with sklearn.

---

## Block 1: Load and inspect (cells 3-4)

**What it does:** loads the Wine dataset into a DataFrame, prints shape (178 x 14) and the class balance:
- class 0: 59 samples (33.1%)
- class 1: 71 samples (39.9%)
- class 2: 48 samples (27.0%)

**Why it matters (theory):**
- The classes are **mildly imbalanced** but not badly. Still, with only 178 samples, a 15% split is just ~27 rows, so a careless split could easily end up with a class nearly missing from validation or test. This motivates stratification in the next block.
- `describe()` shows the features live on wildly different scales (e.g. magnesium around 100, ash around 2.4). Gradient descent struggles when features have very different scales, because the cost surface becomes a stretched valley and one learning rate cannot suit all directions. This motivates standardization.

**Viva line:** "I checked class balance first because it decides whether I need stratified splitting, and the summary statistics show scale differences that force standardization before gradient descent."

---

## Block 2: Stratified split from scratch (cells 5-6)

**What it does:** a function that splits **each class separately** in the 70:15:15 ratio, then combines the pieces. Uses `np.random.default_rng(42)` for reproducibility. Then verifies:
- the three sets are **disjoint** (no shared rows: overlap counts are 0, 0, 0),
- together they cover all 178 rows,
- each split keeps roughly the original class proportions (train: 33/40/27%, val: 32/40/28%, test: 33/40/27%).

**Theory: what is stratified sampling and why?**
- A plain shuffled split treats the dataset as one pool. By chance, a small slice can get too many of one class. Stratification means: shuffle and slice **within each class**, so every split inherits the same class mix by construction.
- Why 70:15:15 and not just train/test? The **validation set** is where we choose hyperparameters (here, the learning rate). If we chose them on the test set, the test score would no longer be an honest estimate of performance on unseen data. Test data must be touched exactly once, at the end.
- The disjointness assertion is a **leakage check**: if the same row sat in both train and test, the model would be evaluated partly on data it saw during training, inflating the score.

**Viva line:** "Stratification guarantees each split preserves the class proportions, which matters because my smallest split is only 25 rows. I also assert the splits are disjoint, which is my proof against data leakage."

---

## Block 3: Standardization, intercept, one-hot encoding (cell 7)

**What it does:**
1. Computes mean and standard deviation of every feature **on the training set only**, then applies the same transformation to train, validation, and test.
2. Adds a column of ones to X (the intercept trick).
3. Converts training labels to **one-hot** vectors (label 0 becomes [1, 0, 0]) but keeps integer labels for evaluation.

**Theory, piece by piece:**

*Standardization (z-score):* each feature becomes (value - mean) / std, so all features have mean 0 and spread 1 on the training data. Two reasons:
- Gradient descent converges much faster and one learning rate works for all features.
- Fitting the statistics on train only avoids **leakage**: validation/test information must never influence anything the model or preprocessing "learns". Val and test are transformed with the train statistics, simulating deployment on truly new data.

*Intercept column of ones:* instead of tracking a separate bias term b, we append a constant feature 1 to every sample. The weight learned for that column IS the bias. This lets every equation be a single clean matrix product X @ Theta.

*One-hot encoding:* the cross-entropy formula needs, for each sample, a vector saying "the correct class gets 1, others get 0". One-hot is exactly that. We keep the integer labels for evaluation because the confusion matrix and accuracy are easier to compute from integers.

*Shapes to remember:* X_train is (123, 14): 123 samples, 13 features + intercept. Theta is (14, 3): one weight column per class. So X @ Theta is (123, 3): one score per class per sample.

---

## Block 4: Softmax, cross-entropy, gradient descent (cells 8-9)

This is the heart of Question 1. Three functions:

### Softmax

For a sample with class scores z = (z_0, z_1, z_2) (these scores are called **logits**):

P(y = k | x) = e^(z_k) / (e^(z_0) + e^(z_1) + e^(z_2))

- Exponentiating makes everything positive; dividing by the sum makes them add to 1. So the output is a valid probability distribution over classes.
- Bigger score means bigger probability, and the mapping is smooth (differentiable), which gradient descent requires.

**The max-subtraction trick:** the code subtracts each row's maximum score before exponentiating. Mathematically nothing changes, because e^(z - m) / sum e^(z_j - m) has the e^(-m) cancel top and bottom. Numerically it matters a lot: e^800 overflows to infinity in floating point, but after subtracting the max, the largest exponent is exactly 0, so overflow is impossible. This is a standard numerical-stability trick, and the viva may ask about it.

### Multiclass cross-entropy cost

J(Theta) = -(1/n) * sum over samples i, sum over classes k of [ Y_ik * log(P_ik) ]

Because Y is one-hot, the inner sum keeps only one term per sample: **minus the log of the probability the model gave to the true class**. So the cost is small when the model puts high probability on correct classes and explodes toward infinity when it is confidently wrong.

**Where does this cost come from? (deep theory)** It is **maximum likelihood estimation**: if we assume the model's probabilities describe how labels are generated, the likelihood of the training labels is the product of P(true class) over all samples. Maximizing that product is the same as minimizing its negative log, which is exactly cross-entropy. So we are not choosing the cost arbitrarily; it is the statistically principled cost for a probabilistic classifier.

**Why clip probabilities inside the log?** If a probability underflows to exactly 0, log(0) is minus infinity and the cost becomes NaN. Clipping to [1e-15, 1 - 1e-15] only inside the log keeps the math finite without altering the model.

### The gradient

gradient = (1/n) * X^T @ (P - Y)

This is remarkably clean: the gradient is the average of (predicted probabilities minus one-hot truth) weighted by the inputs. It has the **same form as binary logistic regression** (X^T (p - y)) and even the same form as linear regression's gradient (X^T (prediction - truth)). This is not a coincidence: sigmoid + binary cross-entropy, softmax + multiclass cross-entropy, and identity + squared error are all "matched pairs" (in statistics: canonical link functions), and matched pairs always give this prediction-minus-truth gradient. If the viva asks "derive the gradient", the honest short answer is: differentiate cross-entropy through the softmax; the softmax's messy derivative cancels perfectly against the log, leaving P - Y.

**Interpretation:** if the model over-predicts class k for a sample (P > Y), the gradient pushes class k's weights down along that sample's features, and vice versa.

### Batch gradient descent and learning-rate selection

- **Batch** means every update uses the entire training set (all 123 rows) to compute the gradient. Fine here because the data is tiny; on big data you would use mini-batches.
- Theta starts at all zeros, and each step is Theta = Theta - lr * gradient. The cost surface of softmax regression is **convex** (bowl-shaped, one global minimum), so gradient descent with a sane learning rate reliably converges; initialization does not trap us in local minima.
- Three learning rates were tried for 3000 iterations each:

| lr | final train CE | validation CE |
|------|------|------|
| 0.01 | 0.0514 | 0.0604 |
| 0.1 | 0.0098 | **0.0566** (best) |
| 1.0 | 0.0013 | 0.0854 |

**How to read this table (important viva point):** lr = 1.0 achieves the LOWEST training cost but the WORST validation cost. Driving training cross-entropy toward 0 means the model is becoming extremely confident, effectively fitting the training set too sharply: overfitting. The selection is made on **validation** cross-entropy, never training, precisely because training cost always rewards more fitting. lr = 0.1 wins.

- Why select using cross-entropy rather than accuracy? Cross-entropy is a finer-grained signal: accuracy only counts right/wrong, cross-entropy also measures **how calibrated the confidence is**. On 25 validation rows, accuracy moves in jumps of 4%, too coarse to distinguish models.

---

## Block 5: Cost curves (cell 10)

**What it does:** plots training cross-entropy vs iteration for all three learning rates, on a log scale.

**What the picture shows:** all three curves decrease monotonically (no divergence, so even lr = 1.0 was not "too big" for stability, just too big for generalization). Larger lr drops faster. The log scale is used because the interesting differences happen near zero, which a linear scale would flatten.

**Viva line:** "Monotonically decreasing cost curves are my evidence that gradient descent converged and the learning rates were stable. The choice among them was made separately, on validation cross-entropy."

---

## Block 6: Softmax reduces to sigmoid when K = 2 (cells 11-12)

**The math (worth being able to reproduce on paper):** with two classes and scores z_0, z_1:

P(class 1) = e^(z1) / (e^(z0) + e^(z1))

Divide numerator and denominator by e^(z1):

P(class 1) = 1 / (1 + e^(z0 - z1)) = 1 / (1 + e^-(z1 - z0)) = **sigmoid(z1 - z0)**

So a two-class softmax only depends on the **difference** of scores. If we fix the class-0 score at 0 (theta_0 = 0), then P(class 1) = sigmoid(z1) exactly: binary logistic regression is a special case of softmax.

**What the code does:** generates 1000 random scores, computes the two-column softmax with the class-0 column set to zero, computes sigmoid of the same scores, and measures the largest difference: about 1.1e-16, which is machine precision (the smallest difference a 64-bit float can even represent). The assert requires it below 1e-12.

**Also worth knowing:** this score-difference fact means softmax has a built-in redundancy: adding any constant vector to all classes' weights leaves the probabilities unchanged (only differences matter). That is why one class's weights can be fixed at zero without losing anything, and it is also why sklearn's older "multinomial" and "one-vs-rest" variants can differ slightly.

---

## Block 7: Manual evaluation on the test set (cells 13-14)

**What it does:** predicts each test sample's class as **argmax** of the predicted probabilities (pick the most probable class), then computes everything by hand:

*Confusion matrix:* a K x K table where entry (row t, column p) counts samples whose true class is t and predicted class is p. Ours is perfectly diagonal: 10, 12, 8, meaning **zero mistakes on all 30 test samples**.

*Accuracy:* fraction of correct predictions = 1.0000.

*Macro precision / recall / F1:* computed **per class from the confusion matrix**, then averaged:
- Precision of class k = of everything predicted as k, how much really was k (TP / (TP + FP), column-wise).
- Recall of class k = of everything truly k, how much we caught (TP / (TP + FN), row-wise).
- F1 = harmonic mean of the two (harmonic, so one bad number drags it down).
- **Macro** averaging means: compute per class, then take the plain mean. Every class counts equally regardless of its size, so a model cannot hide bad performance on the smallest class. (The alternative, micro/weighted averaging, would let the big classes dominate.) With imbalanced classes, macro is the honest choice, which is why the lab asks for it.

*Multiclass log-loss:* the average of -log(probability assigned to the true class) over test samples = 0.0044. This is exactly the cross-entropy cost evaluated on test data. Accuracy says "was the argmax right"; log-loss additionally says "how confident and calibrated were the probabilities". Two models with equal accuracy can have very different log-loss.

**Why 100% accuracy is believable and not a bug:** the Wine classes are chemically well separated after standardization, the test set is only 30 rows, and the split was leakage-checked. Perfect accuracy on a small, easy, properly held-out test set is plausible; the leakage assertions in Block 2 are the defense.

---

## Block 8: Comparison with sklearn (cells 15-16)

**What it does:** trains `sklearn.linear_model.LogisticRegression` on the same standardized training data (dropping our manual ones-column, because sklearn adds its own intercept) and compares test metrics.

**Result:** identical classifications (both 100% accurate), but different log-loss: ours 0.0044, sklearn 0.026.

**The explanation is regularization, and this is the key discussion point:** sklearn's LogisticRegression applies an **L2 penalty by default** (C = 1). The penalty discourages large weights, and large weights are exactly what create extreme, confident probabilities. So sklearn's probabilities are deliberately tempered (closer to uniform), giving a slightly higher log-loss when everything is classified correctly. Our from-scratch model is unpenalized, so it grows more confident and, since it happens to be right on every test sample, scores a lower log-loss.

**The catch (say this in the viva):** lower log-loss here does NOT mean our model is better. On a harder or noisier test set, unregularized over-confidence is punished brutally, because one confidently wrong prediction contributes a huge -log(small probability). Regularized, humbler probabilities are the safer bet in general. The agreement in actual classifications is what validates our implementation.

---

# Question 2: Support Vector Machines

## The problem

Study linear SVMs on synthetic 2D data (where we can SEE the geometry) and on a real dataset (Breast Cancer Wisconsin), focusing on: what are margins and support vectors, and what does the regularization parameter C actually do?

## The core theory (know this cold)

**What an SVM optimizes.** Many straight lines can separate two classes. Logistic regression picks one via probabilities; the SVM picks **the line that maximizes the margin**: the distance from the boundary to the closest training points of each class. Intuition: a boundary with wide clearance on both sides is the most robust to new samples landing near the border.

**Support vectors.** The closest points, the ones sitting on (or violating) the margin, are the support vectors. They alone determine the boundary; every other point could move around (a little) or be deleted without changing the solution at all. This is very different from logistic regression, where every sample pulls on the weights. It also makes the SVM's solution "sparse in samples".

**The decision function.** The model computes f(x) = w · x + b, a signed score. f(x) = 0 is the boundary; f(x) = +1 and f(x) = -1 are the **margin boundaries** (the dashed lines in our plots). The margin width is 2 / ||w||, so maximizing the margin = minimizing ||w||, which makes the connection to regularization: **a small-norm weight vector IS a wide margin**.

**Soft margin and C.** Real data overlaps, so a perfect margin may not exist. The soft-margin SVM minimizes:

(1/2) ||w||^2 + C * (sum of margin violations)

Each violation (a "slack") measures how far a point pokes into the margin or past the boundary. **C is the price of a violation:**
- **Small C**: violations are cheap. The optimizer happily buys a **wide margin** and tolerates points inside it. Many support vectors. Strong regularization. Higher bias, lower variance.
- **Large C**: violations are expensive. The margin **narrows** to avoid them, the boundary bends toward individual borderline points. Few support vectors. Weak regularization. Lower bias, higher variance, risk of overfitting.

**C vs alpha (bridge to Labs 2-3):** in ridge/lasso, alpha multiplies the penalty, so LARGER alpha = MORE regularization. In SVMs, C multiplies the errors instead, so it works inversely: SMALLER C = MORE regularization. Same dial, opposite direction. Expect this as a viva question.

**Hinge loss (if asked what loss the SVM uses):** the violation term is the hinge loss, max(0, 1 - y·f(x)) with labels coded +1/-1. It is zero for points comfortably on the correct side beyond the margin, and grows linearly for points inside the margin or misclassified. Compare with logistic loss, which never reaches exactly zero: the hinge's flat region is why non-support-vector points have zero influence.

---

## Block 9: Plot helper + separable blobs (cells 17-18)

**What it does:** defines a plotting function that draws the data, the decision boundary (solid line, f = 0), both margin boundaries (dashed, f = +1 and f = -1), and circles the support vectors. Then fits `SVC(kernel='linear', C=1)` on two well-separated blobs (make_blobs, seed 42).

**What the picture shows and why it matters:** with clean separation, only a **handful of points** are circled, and they sit exactly on the dashed lines. This is the geometric definition made visible: the boundary is held up entirely by those few boundary points, like a plank balanced on the nearest stones. Everything else is irrelevant to the solution.

**Technical detail worth knowing:** the plot evaluates the model's decision function on a fine grid of points and draws the contour lines at levels -1, 0, +1. That is the standard way to visualize any classifier's boundary.

---

## Block 10: Overlapping blobs, three values of C (cells 19-20)

**What it does:** regenerates blobs with much larger spread (cluster_std = 2.5, so the classes overlap), then fits linear SVMs with C = 0.01, 1, 100 and plots all three side by side, with support-vector counts and training accuracy in the titles.

**What to say about the result:**
- C = 0.01: very wide margin, LOTS of circled points (any point inside the wide margin counts as a support vector). The boundary is governed by the bulk of the data.
- C = 100: narrow margin, few support vectors, the boundary tilts to accommodate individual borderline points.
- On mostly-separable data the boundary itself barely moves; what visibly changes is **margin width and how many points participate**.

**One-line summary for the viva:** "C is the price of a margin violation: cheap violations buy a wide, stable margin; expensive violations force a narrow margin fitted to borderline points."

---

## Block 11: Overlapping data, 80:20 split, C sweep (cells 21-23)

**What it does:** generates a genuinely noisy 2D dataset with `make_classification` (class_sep = 0.8 brings classes close, flip_y = 0.08 randomly flips 8% of labels: irreducible label noise). Splits 80:20 from scratch with seed 42 (with an overlap check). Trains linear SVMs for six C values spanning five orders of magnitude (0.001 to 100), plots each boundary, and tabulates:

| C | support vectors | train acc | val acc |
|------|------|------|------|
| 0.001 | 240 (all!) | 0.854 | 0.800 |
| 0.01 | 178 | 0.862 | 0.800 |
| 0.1 | 109 | 0.854 | **0.833** |
| 1 | 93 | 0.858 | 0.817 |
| 10 | 90 | 0.854 | 0.817 |
| 100 | 90 | 0.854 | 0.817 |

**How to read this table (this is the discussion the lab wants):**
- At C = 0.001 the margin is so wide that **every one of the 240 training points is a support vector**. Maximal regularization.
- As C grows, the support-vector count falls steadily (240 to 90): the model relies on ever fewer, ever more borderline points.
- Training accuracy creeps up slightly with C, but validation accuracy peaks at a **moderate** C = 0.1 and then declines slightly. Why? The 8% flipped labels are pure noise. A large-C model works hard to classify those noisy points, which is fitting noise: **overfitting in miniature**. No boundary can beat the noise floor, so the extra effort only hurts generalization.
- This is the bias-variance trade-off, SVM edition: small C = more bias, more stability; large C = less bias, more variance.

---

## Block 12: Breast cancer data preparation (cell 24)

**What it does:** loads Breast Cancer Wisconsin (569 samples, 30 features, binary target), **relabels so 1 = malignant** (sklearn ships it with 1 = benign), reuses the same `stratified_split` function from Question 1 (70:15:15, seed 42), and standardizes with training statistics only.

**Why relabel?** Metrics like recall are about the **positive class**. With 1 = malignant, recall literally means "fraction of cancers we caught", which is the medically meaningful number (a missed cancer is the costly error). Same convention as Lab 3, which keeps the labs comparable.

**Why standardize for an SVM?** The SVM's margin is measured in Euclidean distance. If one feature has a range of 1000 and another 0.1, distance (and hence the margin and the solution) is dominated by the big-scale feature. Standardization puts all 30 features on an equal footing. Splits check out: 397/84/88 rows, malignant fraction ~0.37 in each (stratification worked).

---

## Block 13: C sweep and model selection (cell 25)

**What it does:** trains linear SVMs for C in {0.001, 0.01, 0.1, 1, 10, 100}, records support-vector count and train/val/test accuracy, then selects C by **validation accuracy**:

| C | SVs | train | val | test |
|------|------|------|------|------|
| 0.001 | 191 | 0.947 | 0.905 | 0.920 |
| 0.01 | 91 | 0.970 | **0.976** | 0.955 |
| 0.1 | 48 | 0.985 | **0.976** | 0.966 |
| 1 | 33 | 0.985 | 0.964 | 0.989 |
| 10 | 27 | 0.990 | 0.952 | 0.955 |
| 100 | 25 | 0.998 | 0.929 | 0.966 |

Selected: **C = 0.01, with 91 support vectors**.

**Points to defend:**
- The same pattern as the 2D pictures, now in 30 dimensions we cannot draw: support vectors fall from 191 to 25 as C grows, training accuracy climbs to 99.8%, but validation accuracy peaks at moderate C and then degrades. High C memorizes borderline training tumors.
- **The tie-break:** C = 0.01 and C = 0.1 tie on validation accuracy (0.976). We keep the smaller C, i.e. the MORE regularized model. Standard reasoning: when performance is equal, prefer the simpler/more constrained model, since it is the safer generalization bet (an Occam's razor argument).
- Note the test column exists in the table only for the discussion; **the selection used validation only**. Peeking at test accuracy to choose C would contaminate the final estimate. (And notice: C = 1 happened to have the best test accuracy. That is exactly the temptation the protocol forbids: choosing it would be fitting the test set.)

---

## Block 14: Final test evaluation (cells 26-27)

**What it does:** evaluates the selected model (C = 0.01) on the untouched test set, with binary metrics computed manually from TP/FP/FN/TN, plus ROC-AUC computed from **decision-function scores**:

- accuracy 0.9545
- precision 1.0000
- recall 0.8788
- F1 0.9355
- ROC-AUC 0.9994

**How to interpret each number:**
- **Precision = 1.0**: every tumor the model called malignant really was malignant. Zero false alarms.
- **Recall = 0.879**: it caught 87.9% of the malignant tumors, i.e. it missed 4 of 33. The model's errors are all of the "missed cancer" kind, which is the more dangerous kind medically. If asked how to fix that: shift the decision threshold on the decision-function score to favor recall at the cost of precision (the SVM's default threshold is score = 0, but nothing forces us to keep it).
- **F1 = 0.935**: harmonic mean of the two, one balanced summary number.
- **ROC-AUC = 0.9994**: the probability that a randomly chosen malignant tumor gets a HIGHER score than a randomly chosen benign one. 0.9994 means the ranking of tumors by score is almost perfect, so the 4 misses are a threshold issue, not a ranking issue.

**Why the decision function instead of probabilities?** An SVM does not natively output probabilities (it is a geometric method, not a probabilistic one). But its decision function f(x) = w·x + b gives a signed distance to the boundary, and ROC-AUC only needs a **ranking** of samples, which those scores provide. This is exactly why the lab says "computed using the decision function scores".

**Closing discussion (memorize the shape of this argument):** small C = strong regularization = wide margin resting on many support vectors = simple, stable boundary. Large C = weak regularization = narrow margin resting on few borderline points = a boundary contorted around training data. Validation performance peaks at moderate C, mirroring Labs 2 and 3 with alpha (remembering the inversion: small C is like large alpha). Moderate regularization matches or beats the extremes while using a more stable model.

---

# Rapid-fire viva Q&A

**Q: Why softmax and not three separate sigmoids?**
A: Three independent sigmoids give three probabilities that need not sum to 1, so they are not a probability distribution over classes. Softmax couples the classes through the shared denominator, guaranteeing they sum to 1.

**Q: Is softmax regression's cost convex? Why does it matter?**
A: Yes, the cross-entropy of a softmax-linear model is convex in Theta. It means gradient descent has a single global minimum, so zero-initialization and a sane learning rate reliably find the best solution.

**Q: Why subtract the row max in softmax?**
A: Purely numerical. It cancels mathematically but caps the largest exponent at 0, preventing overflow of e^z.

**Q: Why did the largest learning rate give the worst validation loss despite the best training loss?**
A: Driving training loss near zero means extreme confidence, i.e. overfitting; validation loss exposes it. That is why selection is done on validation, never training.

**Q: Why macro averaging?**
A: It computes each metric per class then averages, weighting all classes equally. With class imbalance, it stops the majority class from masking failures on the minority class.

**Q: What is a support vector, in one sentence?**
A: A training point on or inside the margin (or misclassified) that actively determines the boundary; all other points have zero influence on the solution.

**Q: What happens to support vectors as C increases?**
A: Their count decreases: the margin narrows, so fewer points touch or violate it. We saw 240 to 90 on the synthetic data and 191 to 25 on breast cancer.

**Q: SVM vs logistic regression, key differences?**
A: Logistic regression is probabilistic (maximizes likelihood, every point influences the fit, outputs calibrated-ish probabilities). SVM is geometric (maximizes margin, only support vectors matter, outputs distances not probabilities). Both are linear boundaries here, and both regularize: L2 penalty vs margin maximization, which are mathematically close cousins.

**Q: Relationship between C and alpha?**
A: Inverse. Alpha multiplies the penalty (bigger = more regularized); C multiplies the error term (bigger = LESS regularized). C ~ 1/alpha conceptually.

**Q: Why fit the scaler on train only, again?**
A: Test statistics are unavailable at deployment time; using them during training leaks information and makes test scores optimistic. All preprocessing must be learned on train and merely applied elsewhere.

**Q: Your Q1 test accuracy is 100%. Isn't that suspicious?**
A: With a well-separated dataset, only 30 test rows, and split disjointness verified by assertion, it is plausible. The log-loss comparison with sklearn shows both models agree, and the disagreement in log-loss is fully explained by sklearn's default L2 regularization.

**Q: If a doctor said recall 0.879 is too low, what would you change without retraining?**
A: Lower the decision threshold on the SVM's decision-function score (classify as malignant even at slightly negative scores). This trades precision for recall; the near-perfect ROC-AUC (0.9994) says the score ranking supports a much better recall at modest precision cost.
