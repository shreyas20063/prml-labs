# Lab 1 Study Notes (EC24BT033)

Line-by-line understanding of Lab1-EC24BT033.ipynb, for rewriting it from scratch and for the presentation.

## Setup
- `import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns`: arrays/random, dataframes, base plotting, statistical plotting. Standard aliases.
- `pd.read_csv(url)`: read_csv accepts a URL directly, so the notebook runs anywhere with no local file.

## Q1
- `df.shape`: (rows, cols) tuple = (1338, 7). N = 1338 samples, d = 6 raw features (charges is the target, not a feature).
- `df.dtypes`: numeric vs text columns. Text columns will need encoding later; d is counted before that.
- `df.head()`: eyeball raw data before statistics.
- `df.describe()`: count/mean/std/min/quartiles/max for numeric columns. count < 1338 would mean missing values.
- Regression because charges is continuous (any positive real value), no finite class set. Polynomial regression is a MODEL choice, not the problem type.
- D = {(x_i, y_i)} for i = 1..1338, x_i in R^6, y_i in R.

## Q2
- `df.groupby(col)["charges"].mean()`: bucket rows by a factor, average the target per bucket. The max-min gap of these means is the "which factor matters" metric.
- Results: smoker gap ~23,616 (32,050 vs 8,434, about 3.8x). Region gap ~2,388. Sex gap ~1,387. Smoker dominates.
- `.corr()`: Pearson correlation matrix. r in [-1, 1], measures LINEAR association only.
- Results: age 0.299 > bmi 0.198 > children 0.068.
- Plots: `plt.subplots(2, 3)` grid; hist (right-skewed charges), boxplot by smoker (box = Q1..Q3, line = median), scatter with hue=smoker (reveals the interaction: obese smokers form a separate 35-60k cluster), heatmap (annot=True prints numbers, vmin/vmax fix the scale), sorted bar chart by region.
- Key insight: bmi's low overall r hides the interaction. Correlation averages a flat non-smoker cloud with a steep smoker cloud. "The plots revealed what the summary numbers hid."

## Q3
- Always `df.copy()` before adding columns, never mutate the original.
- `pd.cut(series, bins=[...], labels=[...])`: bin a continuous column. WHO cutoffs 18.5/25/30 for BMI; np.inf = no upper limit.
- Boolean flags: `((cond1) & (cond2)).astype(int)`. Use `&`/`|`/`~` with parentheses (not and/or). astype(int) makes 1/0.
- Features: bmi_category, age_group, smoker_obese (the Q2 cluster as a flag), family_size (children+1, honestly noted as a near-duplicate of children: a constant shift adds nothing to linear or tree models), smoker_senior (my idea: compounding risk), bmi_x_age (nonlinear numeric interaction).
- No target leakage: nothing is derived from charges.

## Q4
- Signature must match the lab exactly: custom_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42).
- `np.random.default_rng(seed)`: reproducible generator (same seed, same sequence). `rng.permutation(N)`: 0..N-1 shuffled, like MATLAB randperm.
- Sizes: int(0.7*N)=936, int(0.15*N)=200, test takes the remainder (202) so truncation never loses rows. Never compute all three sizes independently.
- Slice the permutation into three contiguous pieces; disjoint by construction.
- `.iloc` = select by position (matches permutation output). `.loc` = by label (bug source here).
- Proof, not promise: sets + intersections. `set(a) & set(b)` empty for all three pairs AND union size == N, wrapped in asserts so future edits fail loudly.
- Bonus k-fold: `np.array_split(perm, k)` gives k nearly-equal chunks; fold i is val, the rest concatenated as train. Every sample validates exactly once. k-fold = better estimates at k times the cost; hold-out = the sets Q6 needs.

## Q5
- Personal corruption seed: 17112006 (stated for reproducibility, makes my corrupted copy unique).
- Corrupt a COPY (df_feat.copy()), never the Q3 dataframe. Separate rng so Q4's sequence is untouched.
- Inject: 3 NaN in bmi + 3 in age (crng.choice with replace=False, then .loc[rows, col] = np.nan), 3 duplicate rows (pd.concat, ignore_index=True to keep labels unique), messy text ('MALE', 'Male ', ' female', 'SOUTHEAST'), one impossible bmi = 250.
- Cleaning ORDER matters: text -> duplicates -> impute -> outliers. Text first so format-only duplicates become exact duplicates.
- Text: `.str.strip().str.lower()` element-wise.
- Duplicates: drop_duplicates found 4, not 3: the original insurance.csv ships with one genuine duplicate row.
- Impute with MEDIAN, not mean: the bmi column still contains 250, the mean would be dragged up and copied into every filled cell; the median is rank-based and immune.
- Outliers with IQR, not z-score: z uses mean/std which the outlier itself inflates (masking effect); quartiles ignore magnitude. Same principle both times: robust statistics cannot be poisoned by the contamination they hunt.
- 1.5xIQR = mild fences (flagged 10 rows, 9 of them real patients with bmi 47-53: plausible severe obesity = signal). 3xIQR = extreme fences (caught only the 250). We drop ONLY extreme. Detection is statistics; removal is a judgment about what could plausibly occur.
- Result: 1341 -> 1336 rows (4 dups + 1 outlier), clean text, no NaN.

## Q6
- z-score chosen over min-max: not defined by the two most extreme samples, gives mean 0/std 1, suits gradient methods and regularization.
- `StandardScaler().fit(train_df[["age","bmi"]])`: THE key line. mu and sigma computed from TRAIN ONLY.
- `.transform` applies train's mu/sigma to train, val, and test alike. charges is never scaled (it is y, not an input).
- Receipts: train is exactly mean 0/std 1 after scaling; val/test are only NEAR 0/1 (e.g. -0.06, 1.01) because they were scaled with train's parameters. That deviation is visible proof of no leakage.
- Why not fit on the full dataset: the test set simulates future unseen data whose statistics we could not have known. Fitting on everything lets test rows influence the preprocessing of training examples (leakage), inflating measured performance; the optimism vanishes in deployment. Same logic: tune on validation, touch test once at the end.
- Histogram before vs after scaling: identical shape, relabeled axis. Z-scoring is linear (shift, divide), it moves the ruler, not the data.

## The one-line thesis for the presentation
Every question was secretly about data leakage: features (Q3, nothing from charges), splits (Q4, proven disjoint), preprocessing (Q6, fit on train only).

## Patterns worth memorizing
1. `groupby(col)[target].mean()` for any "compare groups" question.
2. Boolean masks with `&`, `|`, `~` for any "select rows" question.
3. Fit on train, transform everywhere, for any preprocessing question.

Vocabulary: permutation (shuffle), partition (disjoint + complete), robust statistic (median, IQR), interaction (one feature's effect depends on another), leakage (test information influencing training).
