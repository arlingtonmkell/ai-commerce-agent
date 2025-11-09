"""
tests/benchmark_recommend.py
----------------------------
Rough timing test for recommendation latency.
"""

import time
from recommender.recommend import recommend_products

def test_latency_1k():
    # Warm-up: load model, cache, etc.
    _ = recommend_products("warm up query", top_k=5)

    start = time.time()
    _ = recommend_products("recommend me a backpack", top_k=5)
    dur = (time.time() - start) * 1000
    print(f"⏱️ Query latency: {dur:.1f} ms (post-warmup)")
    assert dur < 200, "Recommendation too slow (>200 ms) after warmup"
