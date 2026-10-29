# -*- coding: utf-8 -*-
"""Оценка: различают ли сигналы проекта правду и ложь на ВАШИХ размеченных данных.

Запуск:
    python evaluate_lie_signals.py                      # все results/*/analysis_log.csv
    python evaluate_lie_signals.py results/*/analysis_log.csv

Как это устроено
* Единица анализа - ФРАГМЕНТ (непрерывный отрезок с одной меткой), а не строка
  журнала. Строки внутри одного фрагмента зависимы, и если считать их
  отдельно, любая случайность выглядит "значимой".
* Показатели внутри фрагмента усредняются с учётом времени.
* Если участников несколько, каждый показатель нормируется внутри участника
  (у каждого человека своя норма моргания, мимики и т.д.).
* Для каждого показателя: AUC (0.5 = случайность), 95% доверительный интервал
  (бутстреп по фрагментам), d Коэна, p-значение (перестановочный тест) и
  p с поправкой на число проверенных показателей.
"""
import csv
import glob
import math
import os
import sys
from datetime import datetime

import numpy as np

from app_paths import RESULTS_DIR

TRUTH, LIE = "правда", "ложь"
MIN_SEG_SECONDS = 3.0     # фрагменты короче не учитываются
MIN_BLINK_SPAN = 4.0      # для частоты морганий нужен фрагмент не короче
MAX_ROW_WEIGHT = 3.0      # потолок "веса" одной строки журнала, с
BOOTSTRAP = 2000
PERMUTATIONS = 5000
MIN_PER_CLASS = 10        # рекомендуемый минимум фрагментов каждого класса

MEAN_FEATURES = [
    ("напряжение_общее", "Индикатор напряжения (общий)"),
    ("напряжение_мимика", "Напряжение: мимика"),
    ("напряжение_моргание", "Напряжение: моргание"),
    ("напряжение_взгляд", "Напряжение: взгляд"),
    ("напряжение_речь", "Напряжение: речь"),
    ("покраснение_общее", "Покраснение (общее)"),
    ("улыбка", "Улыбка"),
    ("рот_открыт", "Рот открыт"),
    ("брови_вверх", "Брови вверх"),
    ("нахмуривание", "Нахмуривание"),
    ("прищур", "Прищур"),
]
DERIVED = [
    ("частота_морганий", "Частота морганий, 1/мин"),
    ("доля_взгляда_не_в_центр", "Доля взгляда не в центр"),
    ("доля_микровыражений", "Доля времени с микровыражением"),
]
FEATURES = MEAN_FEATURES + DERIVED
NAN = float("nan")


def num(value):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return NAN


# ---------------------------------------------------------------- данные

def load_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def build_segments(rows, session):
    times = [num(r.get("время_видео_с")) for r in rows]
    weights = []
    for i, t in enumerate(times):
        nxt = times[i + 1] if i + 1 < len(times) else t + 0.7
        weights.append(min(max(nxt - t, 0.0), MAX_ROW_WEIGHT) if not math.isnan(nxt) else 0.0)

    segments, current = [], None
    for row, t, w in zip(rows, times, weights):
        label = (row.get("метка") or "").strip().lower()
        if label not in (TRUTH, LIE):
            current = None
            continue
        if current is None or current["label"] != label:
            current = {"label": label, "session": session, "rows": [],
                       "participant": (row.get("участник") or "").strip() or session}
            segments.append(current)

        valid = (row.get("лицо_найдено") == "да"
                 and row.get("напряжение_статус") != "калибровка"
                 and not math.isnan(t))
        if valid:
            current["rows"].append((t, w, row))
    return segments


def summarize(seg):
    rows = seg["rows"]
    total = sum(w for _, w, _ in rows)
    if total < MIN_SEG_SECONDS:
        return None

    feats = {}
    for col, _ in MEAN_FEATURES:
        acc = den = 0.0
        for _, w, row in rows:
            v = num(row.get(col))
            if not math.isnan(v):
                acc += v * w
                den += w
        feats[col] = acc / den if den > 0 else NAN

    first, last = rows[0], rows[-1]
    span = last[0] - first[0]
    b0, b1 = num(first[2].get("моргания")), num(last[2].get("моргания"))
    feats["частота_морганий"] = ((b1 - b0) / span * 60.0
                                 if span >= MIN_BLINK_SPAN and not math.isnan(b0 + b1) else NAN)

    off = sum(w for _, w, r in rows if r.get("взгляд") not in ("Центр", "Не найдено"))
    feats["доля_взгляда_не_в_центр"] = off / total
    micro = sum(w for _, w, r in rows if (r.get("микровыражение") or "нет") != "нет")
    feats["доля_микровыражений"] = micro / total

    return {"label": seg["label"], "session": seg["session"],
            "participant": seg["participant"], "seconds": total, "raw": feats}


def add_normalized(segments):
    """Нормировка внутри участника, если участников больше одного."""
    participants = {s["participant"] for s in segments}
    for s in segments:
        s["z"] = dict(s["raw"])
    if len(participants) <= 1:
        return

    for person in participants:
        group = [s for s in segments if s["participant"] == person]
        for name, _ in FEATURES:
            vals = np.array([s["raw"][name] for s in group if not math.isnan(s["raw"][name])])
            ok = len(vals) >= 3 and vals.std() > 0
            for s in group:
                v = s["raw"][name]
                s["z"][name] = (v - vals.mean()) / vals.std() if ok and not math.isnan(v) else NAN


# ---------------------------------------------------------------- статистика

def auc(lie, truth):
    diff = np.asarray(lie, float)[:, None] - np.asarray(truth, float)[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def cohen_d(lie, truth):
    n1, n2 = len(lie), len(truth)
    v1, v2 = np.var(lie, ddof=1), np.var(truth, ddof=1)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    return float((np.mean(lie) - np.mean(truth)) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(lie, truth, rng):
    vals = [auc(rng.choice(lie, len(lie)), rng.choice(truth, len(truth)))
            for _ in range(BOOTSTRAP)]
    return np.percentile(vals, [2.5, 97.5])


def permutation_p(lie, truth, rng):
    observed = abs(auc(lie, truth) - 0.5)
    pool = np.concatenate([lie, truth])
    n = len(lie)
    hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        if abs(auc(pool[:n], pool[n:]) - 0.5) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (PERMUTATIONS + 1)


def evaluate(segments):
    rng = np.random.default_rng(12345)
    results = []
    for name, title in FEATURES:
        lie = np.array([s["z"][name] for s in segments
                        if s["label"] == LIE and not math.isnan(s["z"][name])])
        truth = np.array([s["z"][name] for s in segments
                          if s["label"] == TRUTH and not math.isnan(s["z"][name])])
        if len(lie) < 2 or len(truth) < 2:
            results.append({"title": title, "n": (len(truth), len(lie)), "ok": False})
            continue
        low, high = bootstrap_ci(lie, truth, rng)
        results.append({
            "title": title, "n": (len(truth), len(lie)), "ok": True,
            "auc": auc(lie, truth), "ci": (float(low), float(high)),
            "d": cohen_d(lie, truth), "p": permutation_p(lie, truth, rng),
        })
    k = sum(1 for r in results if r["ok"])
    for r in results:
        if r["ok"]:
            r["p_adj"] = min(r["p"] * k, 1.0)
    return results


def conclusion(r):
    if not r["ok"]:
        return "мало данных"
    excludes_chance = r["ci"][0] > 0.5 or r["ci"][1] < 0.5
    if r["p_adj"] < 0.05 and excludes_chance:
        return "различие есть - проверьте на новых людях"
    if r["p"] < 0.05:
        return "не устояло после поправки"
    return "неотличимо от случайного"


# ---------------------------------------------------------------- отчёт

def build_report(paths, segments, results):
    truth = [s for s in segments if s["label"] == TRUTH]
    lie = [s for s in segments if s["label"] == LIE]
    people = sorted({s["participant"] for s in segments})

    lines = [
        "ОЦЕНКА: различают ли сигналы проекта правду и ложь",
        f"Дата: {datetime.now():%Y-%m-%d %H:%M}",
        f"Сеансов: {len(paths)}, участников: {len(people)}",
        f"Фрагментов: правда - {len(truth)} ({sum(s['seconds'] for s in truth):.0f} c), "
        f"ложь - {len(lie)} ({sum(s['seconds'] for s in lie):.0f} c)",
        "",
    ]
    if min(len(truth), len(lie)) < MIN_PER_CLASS or len(people) < 3:
        lines += [
            f"ВНИМАНИЕ: данных мало (нужно хотя бы {MIN_PER_CLASS} фрагментов каждого класса "
            "и несколько участников). Результаты ниже ненадёжны.",
            "",
        ]

    lines.append(f"{'Показатель':<36}{'n прав/лож':>11}{'AUC':>7}{'95% ДИ':>14}"
                 f"{'d':>7}{'p':>8}{'p попр.':>9}  Вывод")
    for r in results:
        n = f"{r['n'][0]}/{r['n'][1]}"
        if not r["ok"]:
            lines.append(f"{r['title']:<36}{n:>11}  {conclusion(r)}")
            continue
        ci = f"{r['ci'][0]:.2f}-{r['ci'][1]:.2f}"
        lines.append(f"{r['title']:<36}{n:>11}{r['auc']:>7.2f}{ci:>14}{r['d']:>7.2f}"
                     f"{r['p']:>8.3f}{r['p_adj']:>9.3f}  {conclusion(r)}")

    lines += [
        "",
        "Как читать:",
        "  AUC = 0.50 - показатель не различает правду и ложь (как подбрасывание монеты).",
        "  AUC > 0.50 - при лжи показатель выше, AUC < 0.50 - при правде выше.",
        "  Для практической пользы нужно хотя бы 0.80-0.90 при узком доверительном интервале;",
        "  значения 0.55-0.65 означают, что решать по такому признаку нельзя.",
        "  'p попр.' - с поправкой на то, что проверено много показателей сразу.",
        "",
        "Ограничения:",
        "  * Ложь по заданию в эксперименте - не то же самое, что реальная ложь с риском.",
        "  * Результат относится к этим людям, камере и освещению.",
        "  * Даже хороший результат не позволяет судить о человеке по одному эпизоду.",
    ]
    return "\n".join(lines)


def main():
    args = sys.argv[1:] or [os.path.join(RESULTS_DIR, "*", "analysis_log.csv")]
    paths = []
    for arg in args:
        found = glob.glob(arg)
        paths.extend(found if found else ([arg] if os.path.exists(arg) else []))
    paths = sorted(set(paths))
    if not paths:
        print("Не найдено ни одного analysis_log.csv. Укажите путь к файлам.")
        return

    segments = []
    for path in paths:
        session = os.path.basename(os.path.dirname(os.path.abspath(path)))
        for seg in build_segments(load_rows(path), session):
            summary = summarize(seg)
            if summary:
                segments.append(summary)

    if not segments:
        print("Размеченных фрагментов нет. Во время записи нажимайте t (правда) и l (ложь),")
        print("после окончания калибровки индикатора напряжения; n - снять метку.")
        return

    add_normalized(segments)
    results = evaluate(segments)
    report = build_report(paths, segments, results)
    print(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"оценка_{stamp}.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    seg_path = os.path.join(RESULTS_DIR, f"фрагменты_{stamp}.csv")
    with open(seg_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["сеанс", "участник", "метка", "секунд"] + [n for n, _ in FEATURES])
        for s in segments:
            writer.writerow([s["session"], s["participant"], s["label"], round(s["seconds"], 1)]
                            + [round(s["raw"][n], 4) if not math.isnan(s["raw"][n]) else ""
                               for n, _ in FEATURES])
    print(f"\nОтчёт: {report_path}\nФрагменты: {seg_path}")


if __name__ == "__main__":
    main()