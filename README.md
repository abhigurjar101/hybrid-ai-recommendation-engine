# Hybrid AI Recommendation Engine 🎯🛍️

> **A production-grade recommendation engine combining Collaborative Filtering (User-Item Matrix Factorization) with Content-Based Metadata Similarity to solve the Cold-Start problem by Abhi Gurjar.**

---

## 💡 What Is This Project? (Explained Like a Human)

If you've ever wondered how Netflix recommends movies you didn't even know existed, or how Amazon suggests the exact tool you need for a weekend project, they rely on recommendation algorithms.

Traditionally, systems used one of two basic methods:
1. **Collaborative Filtering**: Looks at what people like you bought ("People who watched Stranger Things also watched Dark"). Problem: If a brand-new user signs up today, you have zero data on them! The system is completely blind—this is known as the **Cold-Start Problem**.
2. **Content-Based Filtering**: Looks only at item tags ("You like sci-fi movies, here is another sci-fi movie"). Problem: The user gets trapped in a bubble and never discovers anything new.

**This engine combines both into a Hybrid Ensemble.**
- For active users with purchase history, it uses Matrix Cosine Factorization to find peer user affinity.
- For brand new users or newly cataloged products, it automatically falls back to Content-Based tag similarity.
- Result: **Maximum user engagement, high click-through rates (CTR), and zero cold-start failure.**

---

## 🏛️ System Architecture

```
                       ┌───────────────────────────────┐
                       │     User Interaction Data     │
                       └───────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  [ Collaborative Filtering ]                     [ Content-Based Filtering ]
  • Cosine Matrix Factorization                   • Jaccard & Vector Similarity
  • Peer User Affinity (1-alpha)                  • Catalog Tag Profiles (alpha)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                       [ Weighted Ensemble Blender ]
                                       │
                                       ▼
                     [ Ranked Top-N Recommendations ]
```

---

## 🚀 How to Run in 5 Seconds

Zero external setup or complex database required:

```bash
# 1. Clone the repository
git clone https://github.com/abhigurjar101/hybrid-ai-recommendation-engine.git
cd hybrid-ai-recommendation-engine

# 2. Run the recommendation engine
python3 recommender.py
```

### What You Will See:
- Recommendations for existing users with specific technical affinities.
- Real-time resolution of the Cold-Start challenge for a brand-new user with zero prior ratings.
- Transparent score breakdown showing Content Match %, Peer Collaborative %, and Final Hybrid Score.

---

## 📁 Repository Structure

```
hybrid-ai-recommendation-engine/
├── recommender.py          # Standalone hybrid recommendation engine
├── requirements.txt        # Scikit-Learn, Pandas, NumPy, SciPy
└── README.md               # You are here! Simple human documentation
```

---

## 👤 Author

**Abhi Gurjar**  
- Portfolio: [abhigurjar.vercel.app](https://abhigurjar.vercel.app)  
- GitHub: [@abhigurjar101](https://github.com/abhigurjar101)  
- LinkedIn: [Abhi Gurjar](https://in.linkedin.com/in/abhi-gurjar-b13067203)
