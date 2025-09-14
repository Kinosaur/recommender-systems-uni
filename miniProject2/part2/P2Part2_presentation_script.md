# Item-based Nearest Neighbor (Adjusted Cosine) — Presentation Script

Hello, my name is [Your Name]. This is a simple, slide-aligned walkthrough of my item-based nearest neighbor recommender in `miniProject2/part2/main.ipynb`, following the lecturer’s approach.

Use this as speaker notes. It’s short, consistent with the notebook, and references the exact settings I used.

---

## 1) Alternative Approach — Item-based NN (What and Why)

- Main idea: predict a user’s rating for an unseen item by looking at items they already rated and how similar those items are to the target item.
- Offline step: precompute an item–item similarity matrix.
- Online step: for a target user u and item p, compute a weighted average of u’s ratings using item–item similarities.

Slides mapping:
- Cosine similarity measures angle between rating vectors.
- Problem with basic cosine: different user rating levels (bias).
- Solution: adjusted cosine using mean-centered ratings per user.

---

## 2) Data Preparation (Mean-centering)

What I do:
- Load train ratings, auto-detect columns (`user`, `item`, `rating`).
- Convert IDs to strings, ratings to numeric; drop NaNs.
- Compute each user’s average rating and subtract it: rating_centered = rating − user_mean.
- Build the user–item pivot of centered ratings and save it.

Key code (as in the notebook):

```python
INPUT_PATH = "../rating10user91_trainset.csv"
OUTPUT_PATH = "P2Part2_1Profile_Group4.csv"

df = pd.read_csv(INPUT_PATH)
# ...rename to columns: user, item, rating ...
df['user'] = df['user'].astype(str)
df['item'] = df['item'].astype(str)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating']).copy()

user_mean = df.groupby('user')['rating'].mean()
df['user_mean'] = df['user'].map(user_mean)
df['rating_centered'] = df['rating'] - df['user_mean']

pivot = df.pivot_table(index='user', columns='item', values='rating_centered', aggfunc='mean')
pivot.to_csv(OUTPUT_PATH, float_format='%.4f', na_rep='')
```

Why this matches the slides:
- This “preprocessing: adjusting ratings” is exactly the mean-centering used by adjusted cosine.

---

## 3) Model Building — Adjusted Cosine Similarity

Definition from slides:

$$ sim(a,b) = \frac{\sum_{u \in U} (r_{u,a} - \bar r_u)(r_{u,b} - \bar r_u)}{\sqrt{\sum_{u \in U} (r_{u,a} - \bar r_u)^2} \; \sqrt{\sum_{u \in U} (r_{u,b} - \bar r_u)^2}} $$

My settings (lecturer-aligned):
- MIN_OVERLAP = 2 (ignore degenerate sims from a single co-rater)
- No shrinkage (kept simple)

Output: `P2Part2_2Model_Group4.csv` (items × items).

---

## 4) Online Prediction — Weighted Sum over Similar Items

Lecturer’s formula (denominator uses sum of sim, not absolute):

$$ \hat r_{u,p} = \frac{\sum_{i \in \text{RatedItems}(u)} sim(i,p) \cdot r_{u,i}}{\sum_{i \in \text{RatedItems}(u)} sim(i,p)} $$

My neighbor policy (matches “use most similar” in slides):
- INCLUDE_NEGATIVE_SIMS = False (drop negative similarities)
- TOP_K_NEIGHBORS = 50 (cap to top-K per item by signed similarity)
- USE_ABS_IN_DENOM = False (pure lecturer formula)

Key parameters block in the notebook:

```python
USE_ABS_IN_DENOM = False      # denominator = sum(sim)
INCLUDE_NEGATIVE_SIMS = False # only positive, most similar
TOP_K_NEIGHBORS = 50          # cap neighbors per item
TOP_N = 10                    # recommendations per user
CLIP_MIN, CLIP_MAX = 1.0, 10.0
```

Predict-and-rank flow:
1) Precompute item neighbor lists from the similarity matrix (drop self/zeros/negatives, sort desc by sim, apply top-K).
2) For each user, accumulate numerator/denominator from their rated items’ neighbors.
3) If no contributing neighbors, fall back to user mean (or global mean if cold-start).
4) Sort candidates by predicted score and write top-N.

Output: `P2Part2_3Recommendation_Group4.csv`.

---

## 5) RMSE Evaluation (Test Set)

Metric from slides:

$$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \hat x_i)^2 } $$

What I compute:
- For each test row (user, item, actual), reuse `predict_for_user(user)` to get predictions and look up the item.
- If no prediction for an item (e.g., not in sim matrix), fall back to user mean.
- Save all rows and report RMSE plus fallback counts.

Output: `P2Part2_4_RMSE_Group4.csv`.

---

## 6) Results sanity checks

- Similarity stats: count, min/max, fraction negative (expected lower after applying positive-only filter).
- Neighbor examples: top-5 most similar items for a sample item (by signed similarity).
- Coverage: check how many test predictions relied on fallbacks; large fallback count suggests raising K or relaxing filters.

---

## 7) How this follows the lecturer’s slides

- Adjusted cosine similarity: implemented via mean-centered ratings and cosine on co-raters (slides 34–37).
- Prediction formula: weighted sum with sum(sim) in denominator (slide 38–39).
- “Use most similar neighbors”: exclude negative sims and cap top-K (NN neighborhood).
- “Ignore records with few rating”: enforce MIN_OVERLAP = 2 for stability.

---

## 8) How to run

1) Open `miniProject2/part2/main.ipynb` and run cells top-to-bottom.
2) Make sure these files exist one folder up: `rating10user91_trainset.csv`, `rating10user91_testset.csv`.
3) Produced files:
   - `P2Part2_1Profile_Group4.csv` (centered user–item profile)
   - `P2Part2_2Model_Group4.csv` (item–item similarity)
   - `P2Part2_3Recommendation_Group4.csv` (Top-N per user)
   - `P2Part2_4_RMSE_Group4.csv` (per-row predictions for RMSE)

---

## 9) Optional notes for Q&A

- Trade-offs: higher K improves coverage but may add noise; filtering negatives improves precision but can increase fallbacks.
- If many fallbacks: raise TOP_K_NEIGHBORS (e.g., 100) or allow small negative sims experimentally.
- Scaling up: vectorize similarity with matrix ops, store only top-K per item, or use ANN libraries.

---

Replace [Your Name] and you’re ready to present. If you want, I can also export this to slides or add quick plots (error histogram, predicted vs actual).
