"""
Генератор синтетического датасета банковских транзакций для учебной задачи
"поиск мошеннических/подозрительных операций".

Что тут происходит (коротко):
  1. Генерируется популяция клиентов со своими "привычками" (страна, обычные
     часы активности, любимые категории покупок, обычный уровень трат,
     известные устройства).
  2. Для каждого клиента генерируется история легитимных транзакций за год.
  3. У части клиентов (~4%) "угоняют" карту/аккаунт — вставляется короткая
     пачка мошеннических транзакций с типичными для fraud паттернами
     (чужая страна, новое устройство, ночное время, высокорисковые категории,
     всплеск частоты операций).
  4. Считаются два "чистых" (без утечки в будущее) агрегата на клиента:
     средний чек за последние 30 дней и число операций за последний час —
     оба на момент ДО текущей транзакции.
  5. В финальные labels добавляется реалистичный шум (часть мошенничества
     осталась незамеченной банком, часть легитимных операций ошибочно
     помечена как мошенничество клиентом/банком).
  6. Поверх всего накладывается "грязь": пропуски, дубликаты, разные форматы
     дат, суммы то строкой с запятой, то с лишним пробелом, опечатки в
     регистре и т.д. Это НЕ баг генератора — это специально, чтобы датасет
     требовал полноценной предобработки, как и реальные банковские выгрузки.

Датасет отсортирован по времени и НЕ перемешан специально: если тренировать
модель на случайном train/test сплите, будет утечка данных из будущего в
прошлое (temporal leakage). Об этом подробно — в data_dictionary.md.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_CUSTOMERS = 2500
PERIOD_START = pd.Timestamp("2025-09-04")
PERIOD_END = pd.Timestamp("2026-09-03")

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "pharmacy", "transport",
    "utility_payment", "clothing", "electronics", "entertainment",
    "travel", "online_marketplace", "subscription", "atm_withdrawal",
    "p2p_transfer", "jewelry", "gambling", "crypto_exchange",
]
# категории, которые непропорционально часто встречаются в мошеннических
# операциях (быстро обналичиваемые/трудно отменяемые покупки)
HIGH_RISK_CATEGORIES = ["electronics", "crypto_exchange", "gambling",
                         "online_marketplace", "jewelry", "travel"]

COUNTRIES = ["RU", "BY", "KZ", "AM", "TR", "AE", "DE", "US", "CN", "GB"]
COUNTRY_WEIGHTS_HOME = [0.86, 0.05, 0.04, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005]

CHANNELS = ["POS", "online", "mobile_app", "ATM"]


def make_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    customer_id = [f"CUST{100000 + i}" for i in range(n)]
    home_country = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS_HOME)
    # средний чек клиента (лог-нормальное распределение -> реалистичный "длинный хвост")
    avg_spend = rng.lognormal(mean=7.5, sigma=0.9, size=n)  # ~ сотни-тысячи руб
    # "типичный час" активности клиента (кто-то дневной, кто-то вечерний)
    typical_hour = rng.normal(loc=15, scale=4, size=n) % 24
    # дата открытия счёта — где-то за 0.5..10 лет до начала периода наблюдения
    days_before_start = rng.integers(180, 3650, size=n)
    account_open_date = PERIOD_START - pd.to_timedelta(days_before_start, unit="D")
    # число транзакций за год (Пуассон вокруг разного среднего для разных клиентов)
    lam = rng.gamma(shape=4.0, scale=4.0, size=n)  # среднее ~16, но с разбросом
    n_txn = rng.poisson(lam=lam).clip(min=1)
    # известные устройства клиента (1-3 штуки)
    known_devices = [
        [f"DEV-{rng.integers(10**7, 10**8 - 1)}" for _ in range(rng.integers(1, 4))]
        for _ in range(n)
    ]
    # персональные предпочтения по категориям (у каждого клиента свои 3-6 любимых)
    cat_prefs = [
        rng.choice(MERCHANT_CATEGORIES, size=rng.integers(3, 7), replace=False).tolist()
        for _ in range(n)
    ]

    return pd.DataFrame({
        "customer_id": customer_id,
        "home_country": home_country,
        "avg_spend": avg_spend,
        "typical_hour": typical_hour,
        "account_open_date": account_open_date,
        "n_txn": n_txn,
        "known_devices": known_devices,
        "cat_prefs": cat_prefs,
    })


def gen_legit_transactions(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    period_seconds = (PERIOD_END - PERIOD_START).total_seconds()
    for row in customers.itertuples(index=False):
        k = row.n_txn
        # моменты времени: равномерно по периоду, час смещаем к typical_hour клиента
        offsets = rng.uniform(0, period_seconds, size=k)
        ts = PERIOD_START + pd.to_timedelta(offsets, unit="s")
        hour_shift = rng.normal(loc=0, scale=2.5, size=k)
        ts = ts.floor("D") + pd.to_timedelta(
            ((row.typical_hour + hour_shift) % 24) * 3600
            + rng.integers(0, 3600, size=k), unit="s"
        )
        amount = rng.lognormal(mean=np.log(max(row.avg_spend, 50)), sigma=0.6, size=k)
        category = rng.choice(row.cat_prefs, size=k) if len(row.cat_prefs) else rng.choice(MERCHANT_CATEGORIES, size=k)
        channel = rng.choice(CHANNELS, size=k, p=[0.45, 0.30, 0.20, 0.05])
        device = rng.choice(row.known_devices, size=k)
        merchant_country = np.where(rng.uniform(size=k) < 0.03,
                                     rng.choice(COUNTRIES, size=k), row.home_country)

        rows.append(pd.DataFrame({
            "customer_id": row.customer_id,
            "timestamp": ts,
            "customer_home_country": row.home_country,
            "amount": amount,
            "merchant_category": category,
            "merchant_country": merchant_country,
            "channel": channel,
            "device_id": device,
            "account_open_date": row.account_open_date,
            "is_fraud": 0,
        }))
    return pd.concat(rows, ignore_index=True)


def gen_fraud_bursts(rng: np.random.Generator, customers: pd.DataFrame,
                      fraud_share: float = 0.04) -> pd.DataFrame:
    victims = customers.sample(frac=fraud_share, random_state=rng.integers(0, 2**32 - 1))
    rows = []
    for row in victims.itertuples(index=False):
        burst_size = rng.integers(3, 11)
        # инцидент начинается в случайный момент периода (не в самом конце,
        # чтобы вся пачка уместилась)
        start_offset = rng.uniform(0, (PERIOD_END - PERIOD_START).total_seconds() - 3600)
        start_ts = PERIOD_START + pd.to_timedelta(start_offset, unit="s")
        gaps = rng.uniform(20, 600, size=burst_size)  # секунды между операциями внутри пачки
        ts = start_ts + pd.to_timedelta(np.cumsum(gaps), unit="s")
        # ночные часы — типичный признак мошеннической активности
        night_hours = (ts.hour < 6) | (ts.hour >= 23)
        fraud_country = rng.choice([c for c in COUNTRIES if c != row.home_country], size=1)[0]
        # биполярное распределение сумм: "прощупывание карты" мелкими суммами
        # либо попытка быстро вывести крупную сумму
        is_small = rng.uniform(size=burst_size) < 0.55
        amount = np.where(
            is_small,
            rng.uniform(1, 50, size=burst_size),
            rng.lognormal(mean=np.log(max(row.avg_spend, 50)) + 2.0, sigma=0.5, size=burst_size),
        )
        category = rng.choice(HIGH_RISK_CATEGORIES, size=burst_size)
        new_device = f"DEV-{rng.integers(10**7, 10**8 - 1)}"

        rows.append(pd.DataFrame({
            "customer_id": row.customer_id,
            "timestamp": ts,
            "customer_home_country": row.home_country,
            "amount": amount,
            "merchant_category": category,
            "merchant_country": fraud_country,
            "channel": "online",
            "device_id": new_device,
            "account_open_date": row.account_open_date,
            "is_fraud": 1,
        }))
    if not rows:
        return pd.DataFrame(columns=["customer_id", "timestamp", "customer_home_country",
                                      "amount", "merchant_category", "merchant_country",
                                      "channel", "device_id", "account_open_date", "is_fraud"])
    return pd.concat(rows, ignore_index=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Считает customer_avg_amount_30d и customer_txn_count_1h СТРОГО по
    прошлым операциям клиента (без заглядывания в будущее)."""
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    avg_30d = np.empty(len(df))
    cnt_1h = np.empty(len(df))

    for _, idx in df.groupby("customer_id").groups.items():
        idx = list(idx)
        sub = df.loc[idx, ["timestamp", "amount"]]
        ts = sub["timestamp"].values
        amt = sub["amount"].values
        for i in range(len(idx)):
            t0 = ts[i]
            # окно [t0 - 30d, t0) — строго ДО текущей операции
            window_30d = (ts < t0) & (ts >= t0 - np.timedelta64(30, "D"))
            avg_30d[idx[i]] = amt[window_30d].mean() if window_30d.any() else amt[i]
            window_1h = (ts < t0) & (ts >= t0 - np.timedelta64(1, "h"))
            cnt_1h[idx[i]] = window_1h.sum()

    df["customer_avg_amount_30d"] = avg_30d.round(2)
    df["customer_txn_count_1h"] = cnt_1h.astype(int)
    return df


def add_ip_country_and_account_age(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    is_remote = df["channel"].isin(["online", "mobile_app"])
    same_country_ip = rng.uniform(size=len(df)) < 0.9
    ip_country = np.where(same_country_ip, df["customer_home_country"], df["merchant_country"])
    df["ip_country"] = np.where(is_remote, ip_country, np.nan)

    df["account_age_days"] = (df["timestamp"] - df["account_open_date"]).dt.days
    df = df.drop(columns=["account_open_date"])
    return df


def add_label_noise(rng: np.random.Generator, df: pd.DataFrame,
                     miss_rate: float = 0.12, false_alarm_rate: float = 0.003) -> pd.DataFrame:
    """is_fraud_reported — то, что реально "увидел" банк (шумная метка).
    is_fraud остаётся истинным паттерном генерации только для справки и в
    финальный CSV НЕ идёт как отдельная колонка, чтобы не превращать задачу
    в жульничество; используем зашумлённую метку как единственный target."""
    true_fraud = df["is_fraud"].to_numpy().copy()
    reported = true_fraud.copy()

    fraud_idx = np.where(true_fraud == 1)[0]
    miss_idx = rng.choice(fraud_idx, size=int(len(fraud_idx) * miss_rate), replace=False)
    reported[miss_idx] = 0

    legit_idx = np.where(true_fraud == 0)[0]
    false_alarm_idx = rng.choice(legit_idx, size=int(len(legit_idx) * false_alarm_rate), replace=False)
    reported[false_alarm_idx] = 1

    df["is_fraud"] = reported
    return df


def dirty_it_up(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # --- суммы: разные форматы записи, пропуски, опечатки ---
    amount_str = df["amount"].round(2).astype(str)

    def ru_format(x):
        s = f"{x:,.2f}".replace(",", " ").replace(".", ",")
        return s

    ru_fmt_mask = rng.uniform(size=n) < 0.03
    amount_str = amount_str.where(~ru_fmt_mask, df["amount"].apply(ru_format))

    missing_amt_mask = rng.uniform(size=n) < 0.01
    amount_str = amount_str.where(~missing_amt_mask, "N/A")

    sign_typo_mask = rng.uniform(size=n) < 0.005
    amount_str = np.where(sign_typo_mask, "-" + amount_str.astype(str), amount_str)

    fat_finger_mask = rng.uniform(size=n) < 0.003
    fat_finger_vals = (df["amount"] * 100).round(2).astype(str)
    amount_str = pd.Series(amount_str, index=df.index)
    amount_str = amount_str.where(~fat_finger_mask, fat_finger_vals)

    df["amount"] = amount_str

    # --- currency (почти всегда RUB, но не всегда аккуратно записана) ---
    currency = np.full(n, "RUB", dtype=object)
    foreign_mask = df["merchant_country"] != df["customer_home_country"]
    currency[foreign_mask.to_numpy()] = "USD"
    messy_case_mask = rng.uniform(size=n) < 0.02
    currency = np.array([
        c.lower() if messy_case_mask[i] and i % 2 == 0 else (f" {c} " if messy_case_mask[i] else c)
        for i, c in enumerate(currency)
    ], dtype=object)
    df["currency"] = currency

    # --- пропуски в категориальных полях ---
    for col, rate in [("merchant_category", 0.03), ("device_id", 0.04),
                       ("customer_home_country", 0.01)]:
        mask = rng.uniform(size=n) < rate
        df.loc[mask, col] = np.nan

    # --- регистр/пробелы в merchant_category ---
    case_mask = rng.uniform(size=n) < 0.025
    df.loc[case_mask & df["merchant_category"].notna(), "merchant_category"] = (
        df.loc[case_mask & df["merchant_category"].notna(), "merchant_category"]
        .apply(lambda s: f" {s.upper()} " if rng.uniform() < 0.5 else s.capitalize())
    )

    # --- card_last4 ---
    df["card_last4"] = rng.integers(1000, 9999, size=n).astype(str)
    miss_card_mask = rng.uniform(size=n) < 0.01
    df.loc[miss_card_mask, "card_last4"] = np.nan

    # --- timestamp: часть операций записана в другом формате ---
    ts_str = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    alt_fmt_mask = rng.uniform(size=n) < 0.01
    alt_fmt_str = df["timestamp"].dt.strftime("%d.%m.%Y %H:%M")
    ts_str = ts_str.where(~alt_fmt_mask, alt_fmt_str)
    blank_ts_mask = rng.uniform(size=n) < 0.002
    ts_str = ts_str.where(~blank_ts_mask, "")
    df["timestamp"] = ts_str

    # --- transaction_id ---
    df = df.reset_index(drop=True)
    df["transaction_id"] = [f"TXN{20000000 + i}" for i in range(n)]

    # --- дубликаты строк (случайно продублированная отправка) ---
    dup_frac = 0.004
    dup_sample = df.sample(frac=dup_frac, random_state=int(rng.integers(0, 2**32 - 1)))
    df = pd.concat([df, dup_sample], ignore_index=True)

    return df


def main():
    rng = np.random.default_rng(RNG_SEED)

    customers = make_customers(rng, N_CUSTOMERS)
    legit = gen_legit_transactions(rng, customers)
    fraud = gen_fraud_bursts(rng, customers, fraud_share=0.04)

    df = pd.concat([legit, fraud], ignore_index=True)
    df = add_engineered_features(df)
    df = add_ip_country_and_account_age(rng, df)
    df = add_label_noise(rng, df)
    df = dirty_it_up(rng, df)

    cols = [
        "transaction_id", "timestamp", "customer_id", "customer_home_country",
        "card_last4", "amount", "currency", "merchant_category", "merchant_country",
        "channel", "device_id", "ip_country", "account_age_days",
        "customer_avg_amount_30d", "customer_txn_count_1h", "is_fraud",
    ]
    df = df[cols]

    out_path = "bank_transactions.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Строк: {len(df)}")
    print(f"Доля is_fraud=1: {df['is_fraud'].mean():.4%}")
    print(df.isna().mean().round(4))
    print(f"Сохранено в {out_path}")


if __name__ == "__main__":
    main()
