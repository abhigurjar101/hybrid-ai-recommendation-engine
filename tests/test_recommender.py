import math
import unittest
from recommender import CATALOG, USER_RATINGS, HybridRecommender, jaccard_similarity

class RecommendationTests(unittest.TestCase):
    def test_excludes_seen_items_and_orders_scores(self):
        results = HybridRecommender().recommend("User_Alice", 100)
        self.assertEqual(len(results), len(CATALOG) - len(USER_RATINGS["User_Alice"]))
        self.assertFalse(set(USER_RATINGS["User_Alice"]) & {r["item_id"] for r in results})
        scores = [r["hybrid_match_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= score <= 1 for score in scores))

    def test_unknown_user_is_uniform_not_personalized(self):
        results = HybridRecommender().recommend("unknown", 100)
        self.assertEqual({r["hybrid_match_score"] for r in results}, {0.5})
        self.assertEqual(HybridRecommender().recommend("unknown", 0), [])

    def test_validation_and_empty_tags(self):
        self.assertEqual(jaccard_similarity([], []), 0)
        for alpha in [-1, 2, math.nan, math.inf, True]:
            with self.assertRaises(ValueError):
                HybridRecommender(alpha)
        for count in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                HybridRecommender().recommend("User_Alice", count)
