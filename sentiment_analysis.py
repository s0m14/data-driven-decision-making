import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import re
import os

def load_and_merge_data(path_2gis, path_google=None, path_yandex=None):

    df_2gis = pd.read_csv(path_2gis)
    df_2gis = df_2gis[['company', 'author', 'rating', 'date', 'text', 'source']].copy()
    print(f"2GIS: {len(df_2gis)} отзывов")

    frames = [df_2gis]

    if path_google and os.path.exists(path_google):
        g = pd.read_csv(path_google)
        df_google = pd.DataFrame({
            'company': 'Zakazbiletov',
            'author': g.iloc[:, 1],
            'rating': '',
            'date': g.iloc[:, 6],
            'text': g.iloc[:, 7],
            'source': 'google',
        })
        df_google = df_google[df_google['text'].fillna('').str.len() > 5]
        frames.append(df_google)
        print(f"Google: {len(df_google)} отзывов")

    if path_yandex and os.path.exists(path_yandex):
        y = pd.read_csv(path_yandex)
        df_yandex = pd.DataFrame({
            'company': 'Zakazbiletov',
            'author': y['business-review-view__link'],
            'rating': '',
            'date': y['business-review-view__date'],
            'text': y['spoiler-view__text-container'],
            'source': 'yandex',
        })
        df_yandex = df_yandex[df_yandex['text'].fillna('').str.len() > 5]
        frames.append(df_yandex)
        print(f"Yandex: {len(df_yandex)} отзывов")

    df = pd.concat(frames, ignore_index=True)
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip().str.len() > 2].copy()
    print(f"\nОбъединённый датасет: {len(df)} отзывов")
    return df

POSITIVE_WORDS = [
    'спасибо', 'благодар', 'отлично', 'молодец', 'супер', 'класс', 'рекоменд',
    'быстро', 'оперативно', 'помогл', 'решил', 'нравится', 'понравил', 'удобн',
    'хорош', 'прекрасн', 'замечательн', 'великолепн', 'довольн', 'рад', 'рада',
    'лучш', 'восторг', 'приятн', 'комфорт', 'профессионал', 'качествен',
    'рахмет', 'жақсы', 'керемет', 'тамаша', 'көмектес',
    'good', 'great', 'excellent', 'thanks', 'perfect', 'best', 'love', 'amazing',
]

NEGATIVE_WORDS = [
    'ужас', 'кошмар', 'отврат', 'плох', 'мошенни', 'обман', 'развод',
    'не вернул', 'не возврат', 'не работ', 'не отвеча', 'игнор', 'хуже',
    'безобраз', 'отстой', 'жуть', 'кидал', 'кидок', 'украл', 'списал',
    'жалоб', 'недовол', 'разочаров', 'отклонил', 'заблокиров', 'удалил',
    'невозможно', 'проблем', 'не рекоменд', 'не совету', 'закройтесь',
    'жаман', 'нашар',
    'bad', 'terrible', 'awful', 'worst', 'scam', 'fraud', 'horrible',
]

INTENSIFIERS = ['очень', 'крайне', 'абсолютно', 'полный', 'совершенно']

def rating_sentiment(rating):
    try:
        r = float(rating)
        if r >= 4: return 'positive'
        if r <= 2: return 'negative'
        return 'neutral'
    except (ValueError, TypeError):
        return None


def lexicon_sentiment(text):
    text_lower = text.lower()
    pos_score = sum(len(re.findall(w, text_lower)) for w in POSITIVE_WORDS)
    neg_score = sum(len(re.findall(w, text_lower)) for w in NEGATIVE_WORDS)

    intensity = 1 + 0.3 * sum(1 for w in INTENSIFIERS if w in text_lower)
    if text.count('!') > 2:
        intensity *= 1.2

    pos_score *= intensity
    neg_score *= intensity

    negations = len(re.findall(r'\bне\s+\w+', text_lower))
    if negations > 0 and pos_score > neg_score:
        neg_score += negations * 0.5

    if pos_score > neg_score and pos_score > 0:
        return 'positive', pos_score, neg_score
    elif neg_score > pos_score and neg_score > 0:
        return 'negative', pos_score, neg_score
    return 'neutral', pos_score, neg_score


def combined_sentiment(row):
    r_s = row['sentiment_rating']
    l_s = row['sentiment_lexicon']

    if r_s == l_s and r_s is not None: return r_s
    if r_s is not None and r_s != 'neutral': return r_s
    if r_s is None: return l_s
    if r_s == 'neutral' and l_s != 'neutral': return l_s
    return l_s


def apply_sentiment(df):
    """Применяет все 3 метода анализа тональности."""
    df['sentiment_rating'] = df['rating'].apply(rating_sentiment)

    lex = df['text'].apply(lexicon_sentiment)
    df['sentiment_lexicon'] = [r[0] for r in lex]
    df['pos_score'] = [r[1] for r in lex]
    df['neg_score'] = [r[2] for r in lex]

    df['sentiment'] = df.apply(combined_sentiment, axis=1)
    return df


# ════════════════════════════════════════════════
# ЧАСТЬ 3: ОЦЕНКА КАЧЕСТВА
# ════════════════════════════════════════════════

def evaluate_quality(df):
    """Оценивает качество лексического метода vs рейтинг."""
    has_both = df[df['sentiment_rating'].notna()].copy()
    agreement = (has_both['sentiment_rating'] == has_both['sentiment_lexicon']).mean()

    print(f"\n{'=' * 60}")
    print(f"ОЦЕНКА КАЧЕСТВА")
    print(f"{'=' * 60}")
    print(f"Согласованность (rating vs lexicon): {agreement:.1%}")
    print(f"Количество отзывов с обоими методами: {len(has_both)}")

    labels = ['positive', 'neutral', 'negative']
    print("\nClassification Report (Lexicon vs Rating as ground truth):")
    print(classification_report(
        has_both['sentiment_rating'], has_both['sentiment_lexicon'],
        labels=labels, target_names=labels, zero_division=0
    ))
    return has_both, agreement

def create_visualizations(df, output_dir='plots'):
    """Создаёт все визуализации."""
    os.makedirs(output_dir, exist_ok=True)
    colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
    companies = df['company'].unique().tolist()

    # 1. Sentiment by company
    fig, axes = plt.subplots(1, len(companies), figsize=(6 * len(companies), 5))
    if len(companies) == 1: axes = [axes]
    for i, c in enumerate(companies):
        sub = df[df['company'] == c]
        counts = sub['sentiment'].value_counts()
        for label in ['positive', 'neutral', 'negative']:
            val = counts.get(label, 0)
            axes[i].bar(label, val, color=colors[label])
            axes[i].text(label, val + max(counts) * 0.02, str(val), ha='center', fontweight='bold')
        axes[i].set_title(c.replace('_', ' '), fontweight='bold')
    plt.suptitle('Sentiment Distribution by Company', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_sentiment_by_company.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = df['sentiment'].value_counts()
    ax.pie(sc.values, labels=sc.index, colors=[colors[l] for l in sc.index],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 13})
    ax.set_title(f'Overall Sentiment (n={len(df)})', fontsize=15, fontweight='bold')
    plt.savefig(f'{output_dir}/2_sentiment_pie.png', bbox_inches='tight')
    plt.close()

    # 3. Timeline
    df_d = df[df['date'].notna() & (df['date'].astype(str).str.len() >= 7)].copy()
    df_d['month'] = pd.to_datetime(df_d['date'], errors='coerce').dt.to_period('M')
    df_d = df_d.dropna(subset=['month'])

    fig, axes = plt.subplots(len(companies), 1, figsize=(14, 4 * len(companies)))
    if len(companies) == 1: axes = [axes]
    for i, c in enumerate(companies):
        sub = df_d[df_d['company'] == c]
        if len(sub) == 0: continue
        m = sub.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        for s in ['positive', 'neutral', 'negative']:
            if s not in m.columns: m[s] = 0
        m = m.tail(12)
        ms = [str(x) for x in m.index]
        axes[i].bar(ms, m['positive'], color=colors['positive'], label='Positive')
        axes[i].bar(ms, m['neutral'], bottom=m['positive'], color=colors['neutral'], label='Neutral')
        axes[i].bar(ms, m['negative'], bottom=m['positive'] + m['neutral'], color=colors['negative'], label='Negative')
        axes[i].set_title(c.replace('_', ' '), fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(fontsize=9)
    plt.suptitle('Sentiment Dynamics Over Time', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_sentiment_timeline.png', bbox_inches='tight')
    plt.close()

    # 4. Average rating over time
    df_r = df_d.copy()
    df_r['rating_num'] = pd.to_numeric(df_r['rating'], errors='coerce')
    df_r = df_r.dropna(subset=['rating_num'])

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in companies:
        sub = df_r[df_r['company'] == c]
        if len(sub) == 0: continue
        ma = sub.groupby('month')['rating_num'].mean().tail(12)
        ax.plot([str(x) for x in ma.index], ma.values, marker='o', linewidth=2, label=c.replace('_', ' '))
    ax.set_title('Average Rating Over Time', fontsize=15, fontweight='bold')
    ax.set_ylabel('Avg Rating (1-5)')
    ax.set_ylim(0.5, 5.5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_avg_rating_timeline.png', bbox_inches='tight')
    plt.close()

    # 5. Confusion matrix
    hb = df[pd.to_numeric(df['rating'], errors='coerce').notna()].copy()
    hb['sent_r'] = pd.to_numeric(hb['rating']).apply(
        lambda r: 'positive' if r >= 4 else ('negative' if r <= 2 else 'neutral'))
    lbls = ['positive', 'neutral', 'negative']
    cm = confusion_matrix(hb['sent_r'], hb['sentiment_lexicon'], labels=lbls)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Lex:Pos', 'Lex:Neu', 'Lex:Neg'],
                yticklabels=['Rat:Pos', 'Rat:Neu', 'Rat:Neg'], ax=ax)
    ax.set_title('Confusion Matrix: Rating vs Lexicon', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # 6. Rating distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    for j, c in enumerate(companies):
        sub = df_r[df_r['company'] == c]
        rc = sub['rating_num'].value_counts().sort_index()
        ax.bar([x + j * width for x in rc.index], rc.values, width=width,
               label=c.replace('_', ' '), alpha=0.85)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Rating Distribution by Company', fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_rating_distribution.png', bbox_inches='tight')
    plt.close()

    print(f" 6 графиков сохранены в {output_dir}/")




if __name__ == "__main__":
    PATH_2GIS = "2gis_reviews_data/reviews_ALL.csv"
    PATH_GOOGLE = "google.csv"
    PATH_YANDEX = "yandex.csv"

    print("═" * 60)
    print("ЧАСТЬ 1: ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ")
    print("═" * 60)
    df = load_and_merge_data(PATH_2GIS, PATH_GOOGLE, PATH_YANDEX)

    print("\n" + "═" * 60)
    print("ЧАСТЬ 2: АНАЛИЗ ТОНАЛЬНОСТИ")
    print("═" * 60)
    df = apply_sentiment(df)
    print("\nРаспределение тональности:")
    print(df['sentiment'].value_counts())
    print("\nПо компаниям:")
    for c in df['company'].unique():
        print(f"\n  {c}:")
        print(f"  {df[df['company'] == c]['sentiment'].value_counts().to_string()}")

    # 3. Оценка качества
    evaluate_quality(df)

    # 4. Визуализации
    print("\n" + "═" * 60)
    print("ЧАСТЬ 4: ВИЗУАЛИЗАЦИИ")
    print("═" * 60)
    create_visualizations(df)

    # 5. Сохранение
    df.to_csv("reviews_with_sentiment.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ Результат → reviews_with_sentiment.csv ({len(df)} отзывов)")