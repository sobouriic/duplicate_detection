# Usage:
#   python measure_translate_latency_nocsv.py --target en --trials 3
#   python measure_translate_latency_nocsv.py --target fr --trials 5

import argparse
import statistics
import time
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

try:
    from deep_translator import GoogleTranslator
except Exception as e:
    raise SystemExit("Please `pip install deep-translator` in your venv.") from e


@dataclass
class Sample:
    id: str
    lang: str        # metadata; source='auto' is still used
    length_tag: str  # '2s' | '3s' | '4s' | '5s'
    text: str


# Reusable test set: 12 items (2–5 sentences), EN + FR
TEST_SET: List[Sample] = [
    # 2 sentences
    Sample("en_weather_2", "en", "2s", "It might rain later today. Bring a light jacket just in case."),
    Sample("fr_school_2",  "fr", "2s", "La réunion commence à huit heures. N'oublie pas tes notes."),
    Sample("en_food_2",    "en", "2s", "I tried a new bakery. Their croissants were still warm."),
    # 3 sentences
    Sample("fr_trip_3",    "fr", "3s", "Nous partons demain matin. Le train est à 7h15. J'ai imprimé les billets."),
    Sample("en_app_3",     "en", "3s", "I installed the new app yesterday. The onboarding was smooth. I like the dark mode."),
    Sample("fr_health_3",  "fr", "3s", "Je tousse depuis hier. Je vais voir le médecin. J'espère que ce n'est rien de grave."),
    # 4 sentences
    Sample("en_work_4",    "en", "4s", "We shipped the patch today. QA verified the fix. Performance improved slightly. We’ll monitor logs."),
    Sample("fr_event_4",   "fr", "4s", "Le concert était complet. L'ambiance était incroyable. Le chanteur a salué le public. On y retournera."),
    Sample("en_news_4",    "en", "4s", "The city opened a new park. Families were there all afternoon. The playground looks safe. Parking is still limited."),
    # 5 sentences
    Sample("fr_shop_5",    "fr", "5s", "J'ai visité la nouvelle boutique. Les prix sont raisonnables. Le personnel est serviable. La sélection est variée. Je recommande d'y aller tôt."),
    Sample("en_travel_5",  "en", "5s", "We booked a weekend trip. The hotel is near the station. Breakfast is included. The reviews seem positive. We’ll share photos after."),
    Sample("fr_class_5",   "fr", "5s", "La classe a bien participé. Les exercices ont été compris. Nous avons révisé les notions clés. Le devoir est pour lundi. Je suis confiante pour la suite."),
]


def pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    # nearest-rank on 0..n-1 index space
    idx = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
    return vs[idx]


def translate_once(text: str, target: str) -> Tuple[bool, float, str]:
    start = time.perf_counter()
    try:
        out = GoogleTranslator(source="auto", target=target).translate(text)
        dt = time.perf_counter() - start
        return True, dt, out
    except Exception as e:
        dt = time.perf_counter() - start
        return False, dt, f"ERROR: {e}"


def main():
    ap = argparse.ArgumentParser(description="Measure Google Translate latency .")
    ap.add_argument("--target", default="en", help="Target language code, e.g., en or fr (default: en)")
    ap.add_argument("--trials", type=int, default=3, help="Trials per sample (default: 3)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle item order each trial")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between calls (seconds)")
    args = ap.parse_args()

    rng = random.Random(42)

    # Warm-up one tiny call (establish session, DNS, etc.)
    _ = translate_once("Hello", target=args.target)

    latencies_all: List[float] = []
    latencies_by_len: Dict[str, List[float]] = defaultdict(list)
    errors = 0
    total_calls = 0

    for t in range(args.trials):
        items = list(TEST_SET)
        if args.shuffle:
            rng.shuffle(items)
        for s in items:
            ok, dt, _ = translate_once(s.text, target=args.target)
            total_calls += 1
            if ok:
                latencies_all.append(dt)
                latencies_by_len[s.length_tag].append(dt)
            else:
                errors += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    # Console summary
    print("\n=== Google Translate latency (console-only) ===")
    print(f"Target={args.target}  Trials/item={args.trials}  Total calls={total_calls}  Errors={errors}\n")

    if latencies_all:
        print("Overall:")
        print(f"  count={len(latencies_all)} "
              f"mean={statistics.mean(latencies_all):.3f}s "
              f"p50={statistics.median(latencies_all):.3f}s "
              f"p90={pct(latencies_all, 90):.3f}s "
              f"p95={pct(latencies_all, 95):.3f}s "
              f"p99={pct(latencies_all, 99):.3f}s "
              f"min={min(latencies_all):.3f}s "
              f"max={max(latencies_all):.3f}s")
        print()

    for tag in sorted(latencies_by_len.keys()):
        vals = latencies_by_len[tag]
        print(f"Length {tag}:")
        print(f"  count={len(vals)} "
              f"mean={statistics.mean(vals):.3f}s "
              f"p50={statistics.median(vals):.3f}s "
              f"p90={pct(vals, 90):.3f}s "
              f"p95={pct(vals, 95):.3f}s "
              f"p99={pct(vals, 99):.3f}s "
              f"min={min(vals):.3f}s "
              f"max={max(vals):.3f}s")
        print()

    print("Note: Re-run at different times of day to see network variability. Use --shuffle to randomize order per trial.")

if __name__ == "__main__":
    main()
