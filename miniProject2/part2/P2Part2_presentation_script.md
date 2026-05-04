# Item-Based Collaborative Filtering: A Step-by-Step Implementation

Hello, my name is Kino. In this presentation, I will walk you through our implementation of an item-based nearest neighbor recommender system. Our approach directly follows the methodology outlined in the Week 5 and Week 6 lecture slides on Collaborative Filtering by Asst. Prof. Dr. Rachsuda Setthawong.

---

## 1. The Core Concept: Recommending Similar Items
*(Based on Week 5, Slides 4, 30)*

The main idea of item-based collaborative filtering is to recommend items to a user by finding items similar to what they have liked in the past. Instead of matching users, we find relationships between items based on the ratings of the entire user community[cite: 29, 313].

If a user likes Item A, and the system knows that Item A is very similar to Item B, it will recommend Item B.



---

## 2. Step 1: Data Preparation & Mean-Centering
*(Based on Week 5, Slides 35–36)*

**Goal:** To remove individual user rating bias before calculating item similarity. Different users rate on different scales, and mean-centering normalizes this by adjusting the ratings.

**Process:**
1.  Load the training data.
2.  For each user, calculate their average rating across all items they have rated.
3.  Create a "centered" rating by subtracting the user's average from each of their original ratings.
4.  Build a user-item matrix (pivot table) using these new centered ratings. This becomes our user profile for the model.

**Underlying Code:**
```python
# 1. Load and clean the training data
import pandas as pd
df = pd.read_csv("../rating10user91_trainset.csv")
df['user'] = df['user'].astype(str)
df['item'] = df['item'].astype(str)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating']).copy()

# 2. Compute each user's mean rating
user_mean = df.groupby('user')['rating'].mean()
df['user_mean'] = df['user'].map(user_mean)

# 3. Create the mean-centered rating column
df['rating_centered'] = df['rating'] - df['user_mean']

# 4. Create the user-item matrix from centered ratings
pivot = df.pivot_table(index='user', columns='item', values='rating_centered', aggfunc='mean')
pivot.to_csv("P2Part2_1Profile_Group4.csv", float_format='%.4f', na_rep='')
````

---

## 3\. Step 2: Model Building with Adjusted Cosine Similarity

*(Based on Week 5, Slides 33–37)*

**Goal:** To build the core of our model: an item-item similarity matrix. We use the **Adjusted Cosine Similarity** formula, which is equivalent to performing a standard cosine similarity on our mean-centered data.

**Formula:**
$$sim(a,b) = \frac{\sum_{u \in U} (r_{u,a} - \bar r_u)(r_{u,b} - \bar r_u)}{\sqrt{\sum_{u \in U} (r_{u,a} - \bar r_u)^2} \; \sqrt{\sum_{u \in U} (r_{u,b} - \bar r_u)^2}}$$
*(Source: Week 5, Slide 33)*

**Underlying Code:**
This code iterates through every pair of items, finds the users who rated both (co-rated items)[cite: 368], and computes their similarity using the formula above.

```python
import numpy as np
import pandas as pd

# Parameters from the notebook
MIN_OVERLAP = 2
APPLY_SHRINKAGE = False
SHRINKAGE_LAMBDA = 10

# Get items and values from the centered pivot table
centered_pivot = pivot
items = centered_pivot.columns.to_list()
n_items = len(items)
values = centered_pivot.values
sim_mat = np.zeros((n_items, n_items))

# Loop through each unique pair of items
for i in range(n_items):
    vi = values[:, i]
    mask_i = ~np.isnan(vi)
    for j in range(i, n_items):
        if i == j:
            sim_mat[i, i] = 1.0
            continue
            
        vj = values[:, j]
        mask_j = ~np.isnan(vj)
        
        # Find users who rated both items
        common = mask_i & mask_j
        c = int(common.sum())
        
        # Skip if not enough co-raters
        if c < MIN_OVERLAP:
            continue
            
        # Get centered ratings from common users
        vi_c = vi[common]
        vj_c = vj[common]
        
        # Calculate dot product (numerator) and vector norms (denominator)
        num = np.dot(vi_c, vj_c)
        denom_i = np.sqrt(np.dot(vi_c, vi_c))
        denom_j = np.sqrt(np.dot(vj_c, vj_c))
        
        if denom_i == 0 or denom_j == 0:
            continue
            
        raw_sim = num / (denom_i * denom_j)
        
        # Apply optional shrinkage (disabled in our final run)
        sim_val = raw_sim
        if APPLY_SHRINKAGE:
            weight = c / (c + SHRINKAGE_LAMBDA)
            sim_val = weight * raw_sim

        sim_mat[i, j] = sim_val
        sim_mat[j, i] = sim_val

# Save the final similarity matrix to a file
sim_df = pd.DataFrame(sim_mat, index=items, columns=items)
sim_df.to_csv("P2Part2_2Model_Group4.csv", float_format="%.6f")
```

-----

## 4\. Step 3: Prediction Generation with a Weighted Sum

*(Based on Week 5, Slides 38–39)*

**Goal:** To predict a user's rating for an unseen item. We do this by calculating a weighted average of the user's *original* ratings on similar items they have already rated.

**Formula:**
$$\hat r_{u,p} = \frac{\sum_{i \in \text{RatedItems}(u)} sim(i,p) \cdot r_{u,i}}{\sum_{i \in \text{RatedItems}(u)} sim(i,p)}$$
*(Source: Week 5, Slide 37)*

**Key Parameters in Our Code:**

  * `TOP_K_NEIGHBORS = 10`: We use the 10 most similar items (neighbors) for prediction, our optimized value.
  * `INCLUDE_NEGATIVE_SIMS = False`: We only use neighbors with positive similarity, as recommended for finding the "most similar" items.

**Underlying Code:**
The `predict_for_user` function implements the formula above. It finds a user's rated items and uses their neighbors to predict ratings for unrated items.

```python
def predict_for_user(user_id):
    rated_pairs = user_ratings.get(user_id, [])
    if not rated_pairs:
        return [] # Cannot predict for a user with no ratings
    
    rated_items = {it for it, _ in rated_pairs}
    candidates = all_items - rated_items
    
    num = {}   # Numerator accumulator for each candidate item
    denom = {} # Denominator accumulator for each candidate item
    
    # For each item the user has rated...
    for i, r_ui in rated_pairs:
        # Get its pre-computed neighbors
        for j, sim_val in item_neighbors.get(i, []):
            if j in candidates:
                # Accumulate the numerator: sim(i, j) * rating(u, i)
                num[j] = num.get(j, 0.0) + sim_val * r_ui
                # Accumulate the denominator: sim(i, j)
                denom[j] = denom.get(j, 0.0) + sim_val
    
    # Calculate final predictions
    preds = []
    u_mean = user_mean.get(user_id, global_mean) # Fallback rating
    for j in candidates:
        if j in num and denom.get(j, 0) != 0:
            pred = num[j] / denom[j]
        else:
            pred = u_mean # Fallback if no neighbors contributed
        
        # Clip predictions to the valid rating scale (e.g., 1-10)
        preds.append((j, max(CLIP_MIN, min(CLIP_MAX, pred))))
    
    preds.sort(key=lambda x: (-x[1], x[0]))
    return preds
```

-----

## 5\. Step 4: Model Evaluation with RMSE

*(Based on Week 6, Slides 42–43)*

**Goal:** To measure how accurate our predictions are. We use the **Root Mean Square Error (RMSE)**, which calculates the average magnitude of the errors between our predicted ratings and the actual ratings in the test set. A lower RMSE is better.

**Formula:**
$$RMSE = \sqrt{\frac{\sum_{i=1}^N (x_i - \hat x_i)^2}{N}}$$
*(Source: Week 6, Slide 42)*

**Underlying Code:**
This code iterates through each entry in the test set, gets the model's prediction, and calculates the final RMSE.

```python
from math import sqrt

# Get predictions for all users in the test set
test_users = test_df['user'].unique()
all_user_preds = {u: dict(predict_for_user(u)) for u in test_users}

records = []
fallback_count = 0

# For each actual rating in the test data...
for _, row in test_df.iterrows():
    user, item, actual_rating = str(row['user']), str(row['item']), float(row['rating'])
    
    # Look up the pre-computed prediction for this user
    predicted_rating = all_user_preds.get(user, {}).get(item)
    
    # If no prediction, use the user's average rating as a fallback
    if predicted_rating is None:
        predicted_rating = user_mean.get(user, global_mean)
        fallback_count += 1
        
    records.append((user, item, actual_rating, predicted_rating))

# Create a DataFrame and compute RMSE
pred_df = pd.DataFrame(records, columns=["user", "item", "actual_rating", "predicted_rating"])
mse = ((pred_df["actual_rating"] - pred_df["predicted_rating"]) ** 2).mean()
rmse = sqrt(mse)

print(f"Final RMSE on test set: {rmse:.4f}")
print(f"Number of fallback predictions: {fallback_count}")
```

-----

## 6\. Alignment with Lecture Slides: A Summary

Our implementation strictly adheres to the lecturer's methodology:

  * **Adjusted Cosine Similarity**: We use mean-centering before calculating cosine similarity, as required by the slides.
  * **Prediction Formula**: Our code uses the precise weighted-average formula shown on slide 37, with `sum(sim)` in the denominator.
  * **Neighborhood Selection**: We select the Top-K most similar items with positive similarity, following the nearest-neighbor approach.
  * **Stability**: We require a `MIN_OVERLAP` of 2 co-raters to ensure similarity scores are stable, addressing the "too few co-rated items" problem mentioned on slide 23.

## Summary Table: Mapping Implementation to Lecture Slides

| Step | Lecture Slides | Notebook Implementation |
| :--- | :--- | :--- |
| **Core Concept** | Week 5 (4, 30) | Introduction & Motivation |
| **Data Preparation**| Week 5 (35–36) | Mean-Centering & Profiling |
| **Similarity Model**| Week 5 (33–37) | Adjusted Cosine Similarity |
| **Prediction Logic**| Week 5 (37–39) | Weighted Sum Function |
| **Evaluation Metric**| Week 6 (42–43) | RMSE Calculation |

Thank you. I am now ready for any questions.