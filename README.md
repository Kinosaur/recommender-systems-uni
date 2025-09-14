# recommender-systems-uni

Coursework artifacts and mini-projects for a university module on recommender systems. The repository is primarily Jupyter notebooks plus data/outputs used for reports.

## Repository structure (as of main)
- assignments/
  - A2_group4.xlsx
  - G4_assignment2.docx
  - G4_assignment2.pdf
  - Group4_A4.pdf
  - Group4_A6.pdf
  - calculation_w3_updated_version.xlsx
  - research_papers/ (folder)
- miniProject1/
  - Notebooks: Part1_Step4.ipynb, Part1_Step5.ipynb, Part2_Step3.ipynb, Preprocess_RoomAmentities.ipynb, Preprocess_RoomTypes.ipynb
  - Data: HOTEL_OUTDATASET.CSV, UserData.csv, UserDataWithHotelid.csv, Group4_Part1_preprocessed.csv, Group4_Part1_Profile11.csv, Group4_Part1_Recommendation13.csv, Group4_Part1_SimMatrix12.csv, Group4_Part2_Model22.csv, Group4_Part2_Profile21.csv, Group4_Part2_Recommendation23.csv
  - Code: cosine_similarity_matrix_model.py, profiler_alpha.py, compute_jaccard_sim_matrix.py (currently empty)
  - Reports/archives: CSX4207 Group 4 mini-project 1 (1).pdf, RS_MiniProject1.pdf, miniProject1.zip
- miniProject2/
  - Notebook: bookRecSys.ipynb
  - Data: full_book_details.csv, rating10user91_trainset.csv, rating10user91_testset.csv
  - Folders/archives: part1/ (folder), part2/ (folder), part2.zip
  - Report: RS_MiniProject2.pdf
- slides/ (folder)
- .gitignore
- Example_Golf_regression_model_week8.xlsx

If this snapshot looks stale later, update this section to match the current tree.

## Environment
- Python 3.9+ (3.10+ recommended)
- Jupyter Lab or Notebook

Minimal packages likely needed by the notebooks (adjust if they import more):
```bash
pip install jupyter numpy pandas scipy scikit-learn matplotlib seaborn tqdm
```
If any notebook uses extra libraries (e.g., Surprise, implicit), list and pin them here once confirmed.

## Working with the projects

### miniProject1 (hotel recommendation)
- Data files live under miniProject1/ (e.g., HOTEL_OUTDATASET.CSV, UserData*.csv).
- Notebooks:
  - Preprocess_RoomTypes.ipynb and Preprocess_RoomAmentities.ipynb: data cleaning/feature preparation.
  - Part1_Step4.ipynb and Part1_Step5.ipynb: similarity and recommendation steps for Part 1.
  - Part2_Step3.ipynb: steps for Part 2 modeling/evaluation.
- Python scripts:
  - cosine_similarity_matrix_model.py: computes item/user similarity (cosine). Add a short usage example here if it supports CLI args.
  - profiler_alpha.py: lightweight profiling/utilities.
  - compute_jaccard_sim_matrix.py: currently empty — either implement or remove to avoid confusion.

Suggested run path (interactive):
```bash
jupyter lab
# Open the preprocess notebooks first, then Part1_* and Part2_* notebooks.
```
Outputs such as Group4_Part1_SimMatrix12.csv and Group4_Part1_Recommendation13.csv appear to be generated artifacts; consider moving them to a dedicated outputs/ directory or excluding from Git if they can be reproduced.

### miniProject2 (book recommendation)
- Notebook: bookRecSys.ipynb
- Data: full_book_details.csv, rating10user91_trainset.csv, rating10user91_testset.csv
- Auxiliary: part1/ and part2/ folders (empty in Git) and part2.zip archive.

Open the notebook and run cells. If part2.zip contains source code needed by the notebook, extract it into miniProject2/part2/ and commit the code (zip archives are opaque to reviews and versioning).

## Data and outputs
- The repo currently tracks multiple CSV/XLSX/PDF/DOCX files. This makes history heavy and diffs opaque.
- Recommendation:
  - Keep immutable raw data under data/raw/ and gitignore it; check in only tiny samples.
  - Commit code and notebooks; generate outputs to data/processed/ or outputs/ and either gitignore or manage via Git LFS/DVC.
  - Replace zip archives with their extracted, versioned contents.

## Reproducibility
- Set random seeds in notebooks; record versions with `pip freeze` or `conda env export`.
- Document train/test split strategy (temporal or leave-one-out for implicit data) to avoid inflated metrics from random splits.
- Save evaluation summaries (Precision@K, Recall@K, MAP@K, NDCG@K) alongside the code that produced them.

## Housekeeping
- Fill in a LICENSE file (MIT/Apache-2.0/etc.) if this is intended for public reference.
- Complete compute_jaccard_sim_matrix.py or remove it to reduce dead code.
- Extract contents of miniProject2/part2.zip into versioned files; otherwise results aren’t reviewable.
- Consider expanding .gitignore to exclude large raw data and generated CSVs unless strictly necessary to track.

## How to run (quick)
```bash
git clone https://github.com/K Dinosaur/recommender-systems-uni.git
cd recommender-systems-uni
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate.ps1

pip install -U pip
pip install jupyter numpy pandas scipy scikit-learn matplotlib seaborn tqdm

jupyter lab
# Open miniProject1/* or miniProject2/bookRecSys.ipynb and run cells
```

## Caveats (be explicit with readers)
- Some artifacts in repo are results and not code; without the exact environment they may not be reproducible.
- Large PDFs/DOCX/XLSX in Git cannot be diffed; store sources (code/data) rather than outputs where possible.
