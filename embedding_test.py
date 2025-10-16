# multilingual_eval.py
# Usage:
#   python multilingual_eval.py
# What it does:
# - Evaluates multilingual duplicate detection on a small fixed test set:
#   base EN answers, their FR/AR translations, and same-meaning paraphrases.
# - Prints RAW and CAL similarities, and counts FNs/FPs at a given threshold.

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from dupdet.embedder import embed_text_document
from dupdet.calibration import calibrate

# ===== Config =====
THRESHOLD = 0.80  # target threshold (on CAL) as requested
SHOW_TOP = 0      # set >0 to show top-k similar from the pool per variant (optional)

@dataclass
class Case:
    id: str
    base_en: str
    fr: str
    ar: str
    # Same-meaning paraphrases (in EN/FR/AR); can be 0..N items
    variants: List[str]
    # Distractors: different meaning; used to estimate false positives
    distractors: List[str]

CASES: List[Case] = [
    Case(
        id="transport",
        base_en="Public transport is essential for reducing traffic and pollution.",
        fr="Les transports publics sont essentiels pour réduire la circulation et la pollution.",
        ar="النقل العام ضروري لتقليل الازدحام والتلوث.",
        variants=[
            "Good public transit helps cut car traffic and lowers pollution in cities.",
            "Les bus et les métros aident à diminuer les embouteillages et la pollution.",
        ],
        distractors=[
            "Chocolate cake is my favorite dessert.",
            "Le chat dort sur le canapé.",
            "أحب مشاهدة كرة القدم في المساء.",
        ],
    ),
    Case(
        id="education",
        base_en="Teachers need clear curricula and enough resources to support every student.",
        fr="Les enseignants ont besoin de programmes clairs et de ressources suffisantes pour soutenir chaque élève.",
        ar="يحتاج المعلمون إلى مناهج واضحة وموارد كافية لدعم كل طالب.",
        variants=[
            "Clear programs and proper materials help teachers support all learners.",
            "Des programmes clairs et des moyens suffisants soutiennent tous les élèves.",
        ],
        distractors=[
            "The weather was rainy all week.",
            "Je joue au tennis le dimanche matin.",
            "تعرقل حركة المرور بسبب أعمال الطرق.",
        ],
    ),
    Case(
        id="health",
        base_en="Regular exercise and balanced meals can significantly improve overall health.",
        fr="L'exercice régulier et des repas équilibrés peuvent améliorer fortement la santé globale.",
        ar="يمكن أن يحسن التمرين المنتظم والوجبات المتوازنة الصحة بشكل كبير.",
        variants=[
            "Working out consistently and eating well greatly benefits your health.",
            "Faire du sport et manger équilibré améliore la santé.",
        ],
        distractors=[
            "I lost my keys yesterday.",
            "Le film commence à 21 heures.",
            "تفتقر غرفتي إلى الإضاءة الطبيعية.",
        ],
    ),
    Case(
        id="environment",
        base_en="Planting trees in urban areas helps cool neighborhoods and absorb carbon.",
        fr="Planter des arbres en ville aide à rafraîchir les quartiers et à absorber le carbone.",
        ar="يساعد تشجير المناطق الحضرية على تبريد الأحياء وامتصاص الكربون.",
        variants=[
            "More trees in cities reduce heat and capture CO2.",
            "Plus d'arbres en ville refroidissent l'air et stockent du carbone.",
        ],
        distractors=[
            "My computer crashed during the update.",
            "Je cherche un billet de train pas cher.",
            "أريد تعلم العزف على العود.",
        ],
    ),
    Case(
        id="safety",
        base_en="Wearing seat belts and obeying speed limits reduce injuries on the road.",
        fr="Porter la ceinture et respecter les limitations de vitesse réduisent les blessures sur la route.",
        ar="يقلل ارتداء أحزمة الأمان والالتزام بالسرعة من الإصابات على الطرق.",
        variants=[
            "Seatbelts and safe speeds lower the risk of road injuries.",
            "La ceinture et une vitesse adaptée diminuent les accidents.",
        ],
        distractors=[
            "We adopted a puppy last month.",
            "J'aime cuisiner des crêpes le weekend.",
            "المطعم مزدحم جداً الليلة.",
        ],
    ),
]

# ---------- helpers ----------
def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-12)).astype(np.float32)

def cos(a, b) -> float:
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)))

def embed_all(texts: List[str]):
    uniq = list(dict.fromkeys(texts))  # preserve order, dedupe
    vecs = {t: embed_text_document(t) for t in uniq}
    return vecs

# ---------- evaluation ----------
def main():
    pool: List[Tuple[str, str]] = []  # (tag, text)
    for c in CASES:
        pool += [
            (f"{c.id}:base", c.base_en),
            (f"{c.id}:fr", c.fr),
            (f"{c.id}:ar", c.ar),
        ]
        for i, v in enumerate(c.variants):
            pool.append((f"{c.id}:var{i+1}", v))
        for j, d in enumerate(c.distractors):
            pool.append((f"{c.id}:dist{j+1}", d))

    texts = [t for _, t in pool]
    vecs = embed_all(texts)

    tag2text = {tag: txt for tag, txt in pool}
    tag2vec = {tag: vecs[txt] for tag, txt in pool}

    total_same = 0
    fn = 0
    total_dist = 0
    fp = 0

    print(f"\n=== Multilingual duplicate evaluation (THRESHOLD={THRESHOLD:.2f} on CAL) ===\n")

    for c in CASES:
        print(f"--- {c.id} ---")
        base = tag2vec[f"{c.id}:base"]

        checks = [
            ("fr", tag2vec[f"{c.id}:fr"]),
            ("ar", tag2vec[f"{c.id}:ar"]),
        ] + [(f"var{i+1}", tag2vec[f"{c.id}:var{i+1}"]) for i in range(len(c.variants))]

        # expected duplicates
        for name, v in checks:
            raw = cos(base, v)
            cal = calibrate(raw)
            total_same += 1
            ok = cal >= THRESHOLD
            if not ok:
                fn += 1
            print(f"  same-meaning: base ↔ {name:>4}    CAL={cal:0.6f}  {'OK' if ok else 'FN'}")

        # distractors (should be below threshold)
        for j in range(len(c.distractors)):
            v = tag2vec[f"{c.id}:dist{j+1}"]
            raw = cos(base, v)
            cal = calibrate(raw)
            total_dist += 1
            bad = cal >= THRESHOLD
            if bad:
                fp += 1
            print(f"  distractor : base ↔ dist{j+1}    CAL={cal:0.6f}  {'FP' if bad else 'ok'}")

        # Optional: top-k from the whole pool for base
        if SHOW_TOP > 0:
            sims = []
            for tag, _ in pool:
                if tag == f"{c.id}:base":
                    continue
                r = cos(base, tag2vec[tag])
                sims.append((tag, r, calibrate(r)))
            sims.sort(key=lambda x: -x[2])  # sort by CAL desc
            print(f"  top-{SHOW_TOP} by CAL:")
            for tag, r, cal in sims[:SHOW_TOP]:
                print(f"    {tag:18} RAW={r:0.6f} CAL={cal:0.6f}")

        print()

    # Summary
    fn_rate = (fn / total_same) if total_same else 0.0
    fp_rate = (fp / total_dist) if total_dist else 0.0
    print("=== Summary ===")
    print(f"Expected duplicates tested : {total_same}  | False negatives: {fn}  ({fn_rate:.1%})")
    print(f"Distractors tested         : {total_dist}  | False positives: {fp} ({fp_rate:.1%})\n")

if __name__ == "__main__":
    main()
