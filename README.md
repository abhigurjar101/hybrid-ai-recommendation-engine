# Hybrid recommendation baseline

A dependency-free Python demonstration combining content similarity and user-neighborhood collaborative filtering on a synthetic catalog of six courses and four users.

## Run and test

```bash
git clone https://github.com/abhigurjar101/hybrid-ai-recommendation-engine.git
cd hybrid-ai-recommendation-engine
python3 recommender.py
python3 -m unittest discover -s tests -v
```

No third-party packages are needed for the current demo; `requirements.txt` lists optional data-science packages for future experiments.

## Scoring

Content relevance is the maximum rating-weighted Jaccard overlap between a candidate's tags and previously rated items. Collaborative affinity is a weighted peer rating using cosine similarity over co-rated items. The hybrid score blends both with `alpha`, defaulting to 0.55. When peers provide no rating signal, it uses content only. Seen items are excluded.

Unknown users receive equal 0.5 scores, with catalog order breaking ties. This is a deterministic fallback, not personalized cold-start resolution. Cosine similarity from a single common rating is weak evidence.

## Scope and evaluation

This is neighborhood filtering, not matrix factorization. Scores are ranking heuristics, not click probabilities. No CTR, engagement uplift, offline accuracy, or production-readiness result has been measured. A next step is a train/validation/test split on timestamped interactions and evaluation against popularity using precision@k, recall@k, NDCG@k, and catalog coverage.

`recommender.py` contains the catalog, scoring functions, and CLI. `tests/` checks ranking, seen-item exclusion, cold-start behavior, score bounds, and invalid arguments. GitHub Actions runs the tests on pushes and pull requests.

[Abhi Kumar Gurjar](https://www.linkedin.com/in/abhigurjar) · Data Scientist & GenAI Engineer
