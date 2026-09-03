"""
Быстрая проверка датасета: несколько диагностических графиков на
matplotlib/seaborn. Это НЕ решение домашки — только sanity-check,
что в данных действительно есть сигнал для модели и видно "грязь",
которую придётся чистить.

Парсинг amount/timestamp здесь — минимальный, только чтобы отрисовать
графики. Полноценный pipeline предобработки — часть задания.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLOR_LEGIT = "#4C72B0"   # синий  — легитимные операции
COLOR_FRAUD = "#DD8452"   # оранжевый — мошеннические (colorblind-safe пара)

sns.set_theme(style="whitegrid")


def clean_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_timestamp(series: pd.Series) -> pd.Series:
    # часть строк в ISO, часть в DD.MM.YYYY HH:MM — пробуем оба формата
    iso = pd.to_datetime(series, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    alt = pd.to_datetime(series, format="%d.%m.%Y %H:%M", errors="coerce")
    return iso.fillna(alt)


def main():
    df = pd.read_csv("bank_transactions.csv")
    df["amount_clean"] = clean_amount(df["amount"])
    df["timestamp_clean"] = parse_timestamp(df["timestamp"])
    df["hour"] = df["timestamp_clean"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Баланс классов
    ax = axes[0, 0]
    counts = df["is_fraud"].value_counts().sort_index()
    bars = ax.bar(["Легитимные", "Мошеннические"], counts.values,
                   color=[COLOR_LEGIT, COLOR_FRAUD])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v/len(df):.2%})",
                 ha="center", va="bottom", fontsize=9)
    ax.set_title("Дисбаланс классов")
    ax.set_ylabel("количество операций")

    # 2. Распределение сумм (лог-шкала), легит vs мошенничество
    ax = axes[0, 1]
    valid = df["amount_clean"].notna() & (df["amount_clean"] > 0)
    for label, color, name in [(0, COLOR_LEGIT, "легит"), (1, COLOR_FRAUD, "мошенн.")]:
        vals = df.loc[valid & (df["is_fraud"] == label), "amount_clean"]
        ax.hist(np.log10(vals), bins=40, alpha=0.55, color=color, label=name, density=True)
    ax.set_title("Распределение сумм (log10), легит vs мошенничество")
    ax.set_xlabel("log10(сумма, руб)")
    ax.legend()

    # 3. Доля мошенничества по часу суток
    ax = axes[1, 0]
    by_hour = df.dropna(subset=["hour"]).groupby("hour")["is_fraud"].mean()
    ax.bar(by_hour.index, by_hour.values, color=COLOR_FRAUD)
    ax.set_title("Доля мошеннических операций по часу суток")
    ax.set_xlabel("час")
    ax.set_ylabel("доля is_fraud=1")

    # 4. Доля пропусков по колонкам
    ax = axes[1, 1]
    na_rate = df.drop(columns=["amount_clean", "timestamp_clean", "hour"]).isna().mean()
    na_rate = na_rate[na_rate > 0].sort_values()
    ax.barh(na_rate.index, na_rate.values, color=COLOR_LEGIT)
    ax.set_title("Доля пропусков по колонкам")
    ax.set_xlabel("доля NaN")

    fig.suptitle("bank_transactions.csv — быстрая диагностика", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("eda_overview.png", dpi=140)
    print("Сохранено: eda_overview.png")

    print()
    print("customer_txn_count_1h: среднее у fraud =",
          round(df.loc[df.is_fraud == 1, "customer_txn_count_1h"].mean(), 2),
          " / у легит =", round(df.loc[df.is_fraud == 0, "customer_txn_count_1h"].mean(), 3))
    print("Доля online-канала: fraud =",
          round((df.loc[df.is_fraud == 1, "channel"] == "online").mean(), 2),
          " / легит =", round((df.loc[df.is_fraud == 0, "channel"] == "online").mean(), 2))


if __name__ == "__main__":
    main()
