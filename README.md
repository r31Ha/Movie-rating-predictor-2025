# Movie Success Predictor

A machine learning project that predicts IMDb movie ratings (`vote_average`) from movie metadata using regression models. Built with Python, Pandas, Scikit-learn, XGBoost, and Seaborn.

---

## Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.871 | 0.308 |
| Ridge Regression | 0.871 | 0.308 |
| Random Forest v1 | 0.809 | 0.403 |
| Random Forest v2 | 0.737 | 0.504 |
| Gradient Boosting | 0.737 | 0.505 |
| **XGBoost** | **0.708** | **0.542** |

XGBoost achieved the best performance — predicting movie ratings within ~0.71 stars on a 10-point scale, explaining 54% of rating variance from metadata alone.

---

## Project Structure

```
movie-success-predictor/
├── data/
│   └── your_dataset.csv       # Place your dataset here
├── movie_predictor.ipynb      # Main notebook — all phases
└── README.md
```

---

## Dataset

This project uses a custom TMDB-based dataset with ~946k movies. After filtering for movies with at least 10 votes and valid ratings, the modeling dataset contains **92,464 rows**.

Key columns used:

| Column | Description |
|---|---|
| `vote_average` | Target — IMDb-style rating (0–10) |
| `budget` / `revenue` | Financial data (log-transformed) |
| `runtime` | Film length in minutes |
| `popularity` | TMDB popularity score |
| `vote_count` | Number of user ratings |
| `genre_names` | List of genres per film |
| `original_language` | Language of production |
| `release_date` | Used to extract year and month |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/movie-success-predictor.git
cd movie-success-predictor
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyterlab
```

### 3. Add the dataset

Place your CSV file inside the `data/` folder and update the path in the first notebook cell:

```python
df = pd.read_csv('data/your_dataset.csv')
```

### 4. Run the notebook

```bash
jupyter lab
```

Then open `movie_predictor.ipynb` and run **Kernel → Restart & Run All**.

---

## Methodology

### Phase 1 — Load data
Load the CSV and inspect shape, dtypes, and null counts.

### Phase 2 — Clean
- Remove rows with `vote_count < 10` (unreliable ratings)
- Remove rows where `vote_average == 0` (unrated entries)
- Filter `runtime` to 1–300 minutes
- Parse list-string columns (`genre_names`, `origin_country`) with `ast.literal_eval()`
- Extract `release_year` and `release_month` from `release_date`

### Phase 3 — Feature engineering
- Log-transform skewed columns: `budget`, `revenue`, `popularity`, `vote_count`
- One-hot encode top 10 genres
- Add `is_english` and `is_major_language` flags
- Bucket `release_year` into era groups (pre-1950, 1950s–70s, etc.)
- Compute `vote_pop_ratio` and `rev_budget_ratio` as interaction features

### Phase 4 — Train models
- Train/test split: 80/20
- Models: Linear Regression, Ridge, Random Forest (v1 & v2), Gradient Boosting, XGBoost
- Evaluate with RMSE and R²

### Phase 5 — Evaluate & visualize
- Predicted vs actual scatter plot
- Residual plot
- Feature importance chart
- Error distribution histogram
- Model progression comparison chart

---

## Key Findings

`log_vote_count` is the strongest single predictor of rating (importance ~0.25) — films that attract more voters tend to be rated higher, capturing a popularity-quality signal. `genre_Horror` and `runtime` also rank highly, suggesting genre conventions and film length carry meaningful rating signals.

Raw `budget` and `release_year` showed near-zero correlation with rating and were replaced with derived features (`rev_budget_ratio`, era buckets) that performed better.

The remaining ~46% unexplained variance reflects factors absent from metadata: screenplay quality, direction, acting, and critical reception — things no metadata-only model can capture.

---

## Tools

Python · Pandas · NumPy · Scikit-learn · XGBoost · Seaborn · Matplotlib · Jupyter Lab