
# Item-based Nearest Neighbor (Adjusted Cosine) — Step-by-Step Presentation

Hello, my name is [Your Name]. In this presentation, I will walk you through the implementation of an item-based nearest neighbor recommender system, step by step, directly aligned with the lecture slides from Week 5 and Week 6 (Collaborative Filtering, Asst. Prof. Dr. Rachsuda Setthawong).

---


## 1. Main Idea & Motivation (Slides 4, 30)


df = pd.read_csv(INPUT_PATH)
df['user'] = df['user'].astype(str)
df['item'] = df['item'].astype(str)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating']).copy()
df['user_mean'] = df['user'].map(user_mean)
df['rating_centered'] = df['rating'] - df['user_mean']
pivot = df.pivot_table(index='user', columns='item', values='rating_centered', aggfunc='mean')
pivot.to_csv(OUTPUT_PATH, float_format='%.4f', na_rep='')

## 2. Data Preparation (Mean-Centering) (Slides 35–36)
**Goal:** Adjust ratings to remove user bias before computing similarity.

**Steps:**
1. Load the training ratings (`../rating10user91_trainset.csv`).
2. Convert IDs to strings, ratings to numeric, drop NaNs.
3. Compute each user's average rating and subtract it from their ratings (mean-centering).
4. Build a user–item pivot table of centered ratings and save it.

**Code:**
```python
INPUT_PATH = "../rating10user91_trainset.csv"
OUTPUT_PATH = "P2Part2_1Profile_Group4.csv"
df = pd.read_csv(INPUT_PATH)
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
**Why:** This matches the "preprocessing: adjusting ratings" step in the slides, preparing for adjusted cosine similarity.

---

---


## 3. Model Building — Adjusted Cosine Similarity (Slides 33–37)
**Goal:** Compute similarity between items using mean-centered ratings.

**Formula:**
$$ sim(a,b) = \frac{\sum_{u \in U} (r_{u,a} - \bar r_u)(r_{u,b} - \bar r_u)}{\sqrt{\sum_{u \in U} (r_{u,a} - \bar r_u)^2} \; \sqrt{\sum_{u \in U} (r_{u,b} - \bar r_u)^2}} $$

**Steps:**
1. Use the mean-centered pivot table.
2. For each item pair, find users who rated both.
3. Compute numerator and denominator as above.
4. Require at least 2 co-raters (MIN_OVERLAP = 2) for stability.
5. Save the item–item similarity matrix (`P2Part2_2Model_Group4.csv`).

**Code:**
```python
MIN_OVERLAP = 2
for i in range(n_items):
   for j in range(i, n_items):
      # ...existing code...
      if c < MIN_OVERLAP or c == 0:
         continue
      # ...compute adjusted cosine similarity...
```
**Why:** This implements the adjusted cosine similarity as described in the slides.

---


## 4. Online Prediction — Weighted Sum over Similar Items (Slides 38–39)
**Goal:** Predict a user's rating for an unseen item using their ratings and item similarities.

**Formula:**
$$ \hat r_{u,p} = \frac{\sum_{i \in \text{RatedItems}(u)} sim(i,p) \cdot r_{u,i}}{\sum_{i \in \text{RatedItems}(u)} sim(i,p)} $$

**Steps:**
1. For each item, precompute a list of top-K most similar neighbors (positive similarity only, TOP_K_NEIGHBORS = 50).
2. For each user, for each candidate item, accumulate numerator and denominator from rated items' neighbors.
3. If no contributing neighbors, fall back to user mean (or global mean for cold-start).
4. Sort candidates by predicted score and write top-N recommendations per user (`P2Part2_3Recommendation_Group4.csv`).

**Code:**
```python
USE_ABS_IN_DENOM = False      # denominator = sum(sim)
INCLUDE_NEGATIVE_SIMS = False # only positive, most similar
TOP_K_NEIGHBORS = 50          # cap neighbors per item
TOP_N = 10                    # recommendations per user
CLIP_MIN, CLIP_MAX = 1.0, 10.0
# ...existing code for predict_for_user...
```
**Why:** This matches the lecturer's formula and neighbor selection policy.

---


## 5. RMSE Evaluation (Slides 42–43)
**Goal:** Evaluate prediction accuracy using Root Mean Square Error.

**Formula:**
$$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \hat x_i)^2 } $$

**Steps:**
1. For each test row (user, item, actual), use `predict_for_user(user)` to get predictions.
2. If no prediction for an item, fall back to user mean.
3. Save all rows and report RMSE plus fallback counts (`P2Part2_4_RMSE_Group4.csv`).

**Code:**
```python
from math import sqrt
mse = ((pred_df["actual_rating"] - pred_df["predicted_rating"]) ** 2).mean()
rmse = sqrt(mse)
```
**Why:** This is the standard metric for recommender system accuracy.

---


## 6. Results & Sanity Checks (Slides 39, 41)
- Check similarity stats: count, min/max, fraction negative (should be low after filtering).
- Show top-5 most similar items for a sample item.
- Coverage: how many test predictions relied on fallbacks? If high, consider raising K or relaxing filters.

---


## 7. How This Follows the Lecturer’s Slides
- **Adjusted cosine similarity:** Implemented via mean-centering and cosine on co-raters (Slides 34–37).
- **Prediction formula:** Weighted sum with sum(sim) in denominator (Slides 38–39).
- **Neighbor selection:** Exclude negative sims, cap top-K (NN neighborhood).
- **Ignore records with few ratings:** MIN_OVERLAP = 2 for stability.

---


## 8. How to Run (Practical Steps)
1. Open `miniProject2/part2/main.ipynb` and run cells top-to-bottom.
2. Ensure these files exist one folder up: `rating10user91_trainset.csv`, `rating10user91_testset.csv`.
3. Produced files:
   - `P2Part2_1Profile_Group4.csv` (centered user–item profile)
   - `P2Part2_2Model_Group4.csv` (item–item similarity)
   - `P2Part2_3Recommendation_Group4.csv` (Top-N per user)
   - `P2Part2_4_RMSE_Group4.csv` (per-row predictions for RMSE)

---


## 9. Q&A and Troubleshooting
- **Trade-offs:** Higher K improves coverage but may add noise; filtering negatives improves precision but can increase fallbacks.
- **If many fallbacks:** Raise TOP_K_NEIGHBORS (e.g., 100) or allow small negative sims experimentally.
- **Scaling up:** Vectorize similarity with matrix ops, store only top-K per item, or use ANN libraries.

---

## 10. Summary Table: Slide Mapping
| Step | Slide(s) | Notebook Section |
|------|----------|------------------|
| Main Idea | 4, 30 | Introduction |
| Data Prep | 35–36 | Mean-centering |
| Similarity | 33–37 | Adjusted Cosine |
| Prediction | 38–39 | Weighted Sum |
| RMSE Eval | 42–43 | RMSE Calculation |
| Results | 39, 41 | Sanity Checks |

---

Replace [Your Name] with your name before presenting. This script is designed for clear, step-by-step explanation, with direct code and formula references, and is fully aligned with the lecturer’s slides.
