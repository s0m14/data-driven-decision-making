"""
===============================================================================
FairTicket — Assignment 3: Библиометрический анализ
ПОЛНЫЙ СКРИПТ: Сбор → CSV → Очистка (запускать локально)
===============================================================================
Установка зависимостей:
    pip install requests pandas

Запуск:
    python fairticket_bibliometric.py

Результат:
    openalex_data/openalex_combined_raw.json   — сырые JSON данные
    openalex_data/collection_log.json          — лог и статистика сбора
    openalex_data/openalex_raw.csv             — сырая CSV таблица
    openalex_data/openalex_cleaned.csv         — очищенный финальный датасет
    openalex_data/cleaning_report.txt          — отчёт об очистке
===============================================================================

НАУЧНАЯ ТЕМА:
«Механизмы доверия и обнаружения мошенничества на цифровых маркетплейсах
вторичной перепродажи билетов»

(Trust mechanisms and fraud detection in digital secondary ticket marketplaces)

СВЯЗЬ С FAIRTICKET:
FairTicket — верифицированная платформа перепродажи билетов в Казахстане.
Научные решения из статей помогут:
  1) Fraud detection — обнаружение мошеннических объявлений
  2) Dynamic pricing — справедливое ценообразование
  3) Trust & reputation systems — системы доверия на P2P площадках
  4) Identity verification — верификация продавцов (eGov.kz)
  5) Escrow mechanisms — безопасность транзакций (Kaspi.kz)

ПОИСКОВАЯ СТРАТЕГИЯ:
  Ключевые слова:
    Группа A (домен):   "ticket resale", "secondary ticket market",
                        "online marketplace", "e-commerce platform",
                        "peer-to-peer marketplace"
    Группа B (задача):  "fraud detection", "trust", "reputation system",
                        "dynamic pricing", "user verification", "escrow",
                        "consumer protection"
  Временной диапазон: 2015–2025
  API: OpenAlex (https://api.openalex.org)
  Целевой объём: ≥ 1000 уникальных публикаций
===============================================================================
"""

import requests
import json
import time
import os
import csv
import re
from datetime import datetime
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
BASE_URL = "https://api.openalex.org/works"
PER_PAGE = 200
YEAR_FROM = 2015
YEAR_TO = 2025
OUTPUT_DIR = "openalex_data"

SEARCH_QUERIES = [
    ("Q1_ticket_resale",       "secondary ticket market resale",                   200),
    ("Q2_marketplace_fraud",   "online marketplace fraud detection",               200),
    ("Q3_trust_p2p",           "trust reputation system peer-to-peer platform",    200),
    ("Q4_dynamic_pricing",     "dynamic pricing ticket e-commerce",                200),
    ("Q5_scalping",            "ticket scalping price gouging",                    200),
    ("Q6_identity_verify",     "digital identity verification online marketplace", 200),
    ("Q7_escrow",              "escrow payment system online transaction safety",  200),
    ("Q8_consumer_protect",    "consumer protection e-commerce platform fraud",    200),
    ("Q9_platform_trust",      "platform trust online transaction security",       200),
    ("Q10_price_fairness",     "price fairness perception online marketplace",     200),
]


# ═══════════════════════════════════════════════════════════════════════════
# ШАГ 1: СБОР ДАННЫХ ЧЕРЕЗ OPENALEX API
# ═══════════════════════════════════════════════════════════════════════════

def fetch_works(search_query, query_id, max_results):
    """Собирает публикации через OpenAlex API с cursor-пагинацией."""
    all_works = []
    cursor = "*"
    page = 0

    params_base = {
        "search": search_query,
        "filter": f"publication_year:{YEAR_FROM}-{YEAR_TO},type:article|review",
        "per_page": min(PER_PAGE, max_results),
        "mailto": EMAIL,
        "select": ("id,doi,title,publication_year,authorships,"
                   "primary_location,cited_by_count,keywords,concepts,type"),
    }

    while cursor and len(all_works) < max_results:
        params = {**params_base, "cursor": cursor}
        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                print(f"    ⚠ Попытка {attempt+1}/3 не удалась: {e}")
                time.sleep(3 * (attempt + 1))
        else:
            print(f"    ✗ Пропускаем страницу после 3 попыток")
            break

        results = data.get("results", [])
        if not results:
            break

        all_works.extend(results)
        page += 1
        total_avail = data.get("meta", {}).get("count", "?")
        print(f"    Стр. {page}: +{len(results)} (собрано: {len(all_works)}, "
              f"доступно в API: {total_avail})")

        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(0.25)

    return all_works[:max_results]


def collect_all():
    """Основная функция сбора — запускает все запросы, дедуплицирует, сохраняет."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FairTicket — Сбор библиометрических данных (OpenAlex)    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Период: {YEAR_FROM}–{YEAR_TO}")
    print(f"Целевой объём: ≥ 1000 публикаций\n")

    all_works = []
    seen_ids = set()
    stats = {}

    for qid, search, max_res in SEARCH_QUERIES:
        print(f"\n{'─'*60}")
        print(f"  Запрос {qid}: \"{search}\"")
        print(f"{'─'*60}")

        works = fetch_works(search, qid, max_res)

        # Сохраняем сырой JSON по каждому запросу
        with open(os.path.join(OUTPUT_DIR, f"raw_{qid}.json"), "w", encoding="utf-8") as f:
            json.dump(works, f, ensure_ascii=False, indent=2)

        # Дедупликация по OpenAlex ID
        new_count = 0
        for w in works:
            wid = w.get("id", "")
            if wid and wid not in seen_ids:
                seen_ids.add(wid)
                w["_source_query"] = qid
                all_works.append(w)
                new_count += 1

        stats[qid] = {"search": search, "fetched": len(works), "new_unique": new_count}
        print(f"    ✓ Собрано: {len(works)}, новых уникальных: {new_count}")

    # Сохраняем объединённый JSON
    combined_path = os.path.join(OUTPUT_DIR, "openalex_combined_raw.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_works, f, ensure_ascii=False, indent=2)

    # Лог сбора
    log = {
        "timestamp": datetime.now().isoformat(),
        "topic": "Trust mechanisms and fraud detection in digital secondary ticket marketplaces",
        "topic_ru": "Механизмы доверия и обнаружения мошенничества на цифровых маркетплейсах вторичной перепродажи билетов",
        "year_range": f"{YEAR_FROM}-{YEAR_TO}",
        "api": "OpenAlex",
        "queries": stats,
        "total_unique": len(all_works),
    }
    with open(os.path.join(OUTPUT_DIR, "collection_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # Итоговая статистика
    print(f"\n{'═'*60}")
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print(f"{'═'*60}")
    for qid, s in stats.items():
        print(f"  {qid}: собрано {s['fetched']}, уникальных +{s['new_unique']}")
    print(f"{'─'*60}")
    print(f"  ВСЕГО УНИКАЛЬНЫХ: {len(all_works)}")
    print(f"{'═'*60}")

    return all_works


# ═══════════════════════════════════════════════════════════════════════════
# ШАГ 2: ФОРМИРОВАНИЕ CSV ТАБЛИЦЫ
# ═══════════════════════════════════════════════════════════════════════════

def extract_authors(authorships):
    """Извлекает список имён авторов из authorships."""
    names = []
    for a in (authorships or []):
        author = a.get("author", {})
        name = author.get("display_name", "")
        if name:
            names.append(name)
    return "; ".join(names) if names else ""


def extract_journal(primary_location):
    """Извлекает название журнала из primary_location."""
    if not primary_location:
        return ""
    source = primary_location.get("source")
    if source:
        return source.get("display_name", "")
    return ""


def extract_keywords(work):
    """Извлекает ключевые слова из keywords и concepts."""
    kws = set()
    # Из поля keywords (новый формат OpenAlex)
    for kw in (work.get("keywords") or []):
        if isinstance(kw, dict):
            kw_text = kw.get("keyword", kw.get("display_name", ""))
        else:
            kw_text = str(kw)
        if kw_text:
            kws.add(kw_text.strip())
    # Из поля concepts (старый формат, top-5 по score)
    concepts = work.get("concepts") or []
    concepts_sorted = sorted(concepts, key=lambda c: c.get("score", 0), reverse=True)
    for c in concepts_sorted[:5]:
        name = c.get("display_name", "")
        if name:
            kws.add(name.strip())
    return "; ".join(sorted(kws)) if kws else ""


def extract_countries(authorships):
    """Извлекает уникальные страны из аффилиаций авторов."""
    countries = set()
    for a in (authorships or []):
        for inst in (a.get("institutions") or []):
            cc = inst.get("country_code", "")
            if cc:
                countries.add(cc.upper())
    return "; ".join(sorted(countries)) if countries else ""


def json_to_csv(works):
    """Конвертирует JSON данные в CSV таблицу."""
    print("\n\n╔══════════════════════════════════════════════════════════════╗")
    print("║   ШАГ 2: Формирование CSV таблицы                         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    rows = []
    for w in works:
        row = {
            "openalex_id": w.get("id", ""),
            "doi": w.get("doi", ""),
            "title": (w.get("title") or "").strip(),
            "year": w.get("publication_year", ""),
            "authors": extract_authors(w.get("authorships")),
            "journal": extract_journal(w.get("primary_location")),
            "cited_by_count": w.get("cited_by_count", 0),
            "keywords": extract_keywords(w),
            "country": extract_countries(w.get("authorships")),
            "type": w.get("type", ""),
            "_source_query": w.get("_source_query", ""),
        }
        rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, "openalex_raw.csv")
    fieldnames = ["openalex_id", "doi", "title", "year", "authors",
                  "journal", "cited_by_count", "keywords", "country",
                  "type", "_source_query"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ CSV создан: {csv_path}")
    print(f"  ✓ Записей: {len(rows)}")
    print(f"  ✓ Полей: {len(fieldnames)}")
    print(f"  ✓ Поля: {', '.join(fieldnames)}")

    return rows, csv_path


# ═══════════════════════════════════════════════════════════════════════════
# ШАГ 3: ОЧИСТКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════

def clean_data(csv_path):
    """Полная очистка данных с документированием всех изменений."""
    print("\n\n╔══════════════════════════════════════════════════════════════╗")
    print("║   ШАГ 3: Очистка и нормализация данных                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    import pandas as pd

    report_lines = []
    def log(msg):
        print(f"  {msg}")
        report_lines.append(msg)

    # Загрузка
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    initial_count = len(df)
    log(f"Загружено записей: {initial_count}")
    log(f"Столбцы: {list(df.columns)}")
    log("")

    # ─── 3.1 Удаление полных дубликатов ─────────────────────────────────
    log("─── 3.1 Удаление дубликатов ───")
    dup_full = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    log(f"  Полных дубликатов (все поля): {dup_full} → удалены")

    # Дубликаты по openalex_id (уже дедуплицированы при сборе, но проверяем)
    dup_id = df.duplicated(subset=["openalex_id"]).sum()
    df.drop_duplicates(subset=["openalex_id"], keep="first", inplace=True)
    log(f"  Дубликатов по openalex_id: {dup_id} → удалены")

    # Дубликаты по title (нечувствительно к регистру)
    df["_title_lower"] = df["title"].str.lower().str.strip()
    dup_title = df.duplicated(subset=["_title_lower"]).sum()
    df.drop_duplicates(subset=["_title_lower"], keep="first", inplace=True)
    df.drop(columns=["_title_lower"], inplace=True)
    log(f"  Дубликатов по title (lower): {dup_title} → удалены")
    log(f"  Записей после дедупликации: {len(df)}")
    log("")

    # ─── 3.2 Обработка пропусков ────────────────────────────────────────
    log("─── 3.2 Обработка пропусков ───")
    log("  Пропуски до очистки:")
    for col in df.columns:
        nulls = df[col].isna().sum()
        empty_str = (df[col].astype(str).str.strip() == "").sum()
        if nulls > 0 or empty_str > 0:
            log(f"    {col}: NaN={nulls}, пустые строки={empty_str}")

    # Заменяем пустые строки на NaN, чтобы dropna() убрал и их тоже
    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    log(f"  Удалено строк с пропусками (NaN/пустые): {dropped}")
    log(f"  Осталось строк: {len(df)}")

    df["year"] = df["year"].astype(int)
    df["cited_by_count"] = df["cited_by_count"].astype(int)
    log("")

    # ─── 3.3 Нормализация имён авторов ──────────────────────────────────
    log("─── 3.3 Нормализация авторов ───")

    def normalize_author_name(name):
        """Приводит имя автора к формату 'Фамилия И.О.'"""
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)  # множественные пробелы
        # Убираем цифры и спецсимволы кроме точки, дефиса, пробела, апострофа
        name = re.sub(r'[^\w\s\.\-\'\u0400-\u04FF]', '', name)
        # Title case
        parts = name.split()
        normalized = []
        for p in parts:
            if len(p) <= 2 and p.endswith('.'):
                normalized.append(p.upper())
            else:
                normalized.append(p.capitalize())
        return ' '.join(normalized)

    def normalize_authors_field(authors_str):
        if not authors_str:
            return ""
        authors = [normalize_author_name(a) for a in authors_str.split(";")]
        return "; ".join([a for a in authors if a])

    df["authors"] = df["authors"].apply(normalize_authors_field)
    log(f"  Нормализованы имена: удалены лишние пробелы, спецсимволы, Title Case")
    log("")

    # ─── 3.4 Нормализация названий журналов ─────────────────────────────
    log("─── 3.4 Нормализация журналов ───")

    def normalize_journal(j):
        if not j:
            return ""
        j = j.strip()
        j = re.sub(r'\s+', ' ', j)
        # Убираем trailing точку
        j = j.rstrip('.')
        return j

    before_unique = df["journal"].nunique()
    df["journal"] = df["journal"].apply(normalize_journal)
    after_unique = df["journal"].nunique()
    log(f"  Уникальных журналов до: {before_unique}, после: {after_unique}")

    # Показываем пустые
    empty_journal = (df["journal"] == "").sum()
    log(f"  Записей без журнала: {empty_journal}")
    log("")

    # ─── 3.5 Проверка года ──────────────────────────────────────────────
    log("─── 3.5 Проверка диапазона year ───")
    out_of_range = df[(df["year"] < YEAR_FROM) | (df["year"] > YEAR_TO)]
    log(f"  Записей вне {YEAR_FROM}-{YEAR_TO}: {len(out_of_range)}")
    if len(out_of_range) > 0:
        df = df[(df["year"] >= YEAR_FROM) & (df["year"] <= YEAR_TO)]
        log(f"  → Удалены")
    log("")

    # ─── 3.6 Выбросы по cited_by_count (IQR) ───────────────────────────
    log("─── 3.6 Проверка выбросов (cited_by_count, IQR×1.5) ───")
    Q1 = df["cited_by_count"].quantile(0.25)
    Q3 = df["cited_by_count"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    log(f"  Q1={Q1}, Q3={Q3}, IQR={IQR}")
    log(f"  Границы: [{lower}, {upper}]")

    outliers = df[(df["cited_by_count"] < lower) | (df["cited_by_count"] > upper)]
    log(f"  Выбросов: {len(outliers)}")
    log(f"  Решение: оставляем — в библиометрике высокоцитируемые статьи")
    log(f"  информативны. Маркируем флагом is_outlier.")

    df["is_highly_cited"] = (df["cited_by_count"] > upper).astype(int)
    log(f"  Маркировано как highly_cited: {df['is_highly_cited'].sum()}")
    log("")

    # ─── 3.7 Удаление служебных столбцов ────────────────────────────────
    log("─── 3.7 Финализация ───")
    # Убираем _source_query (служебный), оставляем для анализа если нужно
    final_columns = ["openalex_id", "doi", "title", "year", "authors",
                     "journal", "cited_by_count", "keywords", "country",
                     "type", "is_highly_cited"]
    df = df[final_columns].reset_index(drop=True)

    log(f"  Финальных записей: {len(df)}")
    log(f"  Финальных столбцов: {len(df.columns)}")
    log(f"  Столбцы: {list(df.columns)}")
    log("")

    # ─── Статистика финального датасета ──────────────────────────────────
    log("═══ СТАТИСТИКА ФИНАЛЬНОГО ДАТАСЕТА ═══")
    log(f"  Публикаций: {len(df)}")
    log(f"  Диапазон лет: {df['year'].min()} – {df['year'].max()}")
    log(f"  Уникальных журналов: {df[df['journal']!='']['journal'].nunique()}")
    log(f"  Уникальных стран: {len(set(c for cs in df['country'] for c in cs.split('; ') if c))}")
    log(f"  Медиана цитирований: {df['cited_by_count'].median()}")
    log(f"  Среднее цитирований: {df['cited_by_count'].mean():.1f}")
    log(f"  Макс. цитирований: {df['cited_by_count'].max()}")
    log(f"  Высокоцитируемые (outliers): {df['is_highly_cited'].sum()}")

    # Топ-5 по годам
    log("\n  Публикации по годам:")
    for year, cnt in df['year'].value_counts().sort_index().items():
        log(f"    {year}: {cnt}")

    # ─── Сохранение ─────────────────────────────────────────────────────
    cleaned_path = os.path.join(OUTPUT_DIR, "openalex_cleaned.csv")
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    log(f"\n  ✓ Финальный датасет: {cleaned_path}")

    # Отчёт об очистке
    report_path = os.path.join(OUTPUT_DIR, "cleaning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FairTicket — Отчёт об очистке библиометрических данных\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Исходных записей: {initial_count}\n")
        f.write(f"Финальных записей: {len(df)}\n")
        f.write(f"Удалено: {initial_count - len(df)}\n\n")
        for line in report_lines:
            f.write(line + "\n")
    log(f"  ✓ Отчёт: {report_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Шаг 1: Сбор из OpenAlex
    works = collect_all()

    if len(works) == 0:
        print("\n⚠ Не удалось собрать данные. Проверьте интернет-соединение.")
        exit(1)

    # Шаг 2: JSON → CSV
    rows, csv_path = json_to_csv(works)

    # Шаг 3: Очистка
    df_clean = clean_data(csv_path)

    print("\n\n╔══════════════════════════════════════════════════════════════╗")
    print("║   ✅ ГОТОВО! Все файлы в папке openalex_data/              ║")
    print("╚══════════════════════════════════════════════════════════════╝")