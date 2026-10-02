import pandas as pd
import os


def load_colors():

    possible_paths = [
        "data/colors.csv",
        "colors.csv"
    ]

    for path in possible_paths:

        if os.path.exists(path):
            return pd.read_csv(path)

    return pd.DataFrame(
        columns=["Color", "HEX", "Category"]
    )


def recommend_colors(room_type, category="All", number=6):

    df = load_colors()

    if df.empty:
        return df

    if category != "All" and "Category" in df.columns:

        filtered = df[
            df["Category"].str.lower()
            == category.lower()
        ]

    else:
        filtered = df

    if len(filtered) < number:
        number = len(filtered)

    return filtered.sample(
        n=number,
        random_state=42
    )


def get_color_by_name(color_name):

    df = load_colors()

    if df.empty:
        return None

    result = df[
        df["Color"].str.lower()
        == color_name.lower()
    ]

    if len(result) > 0:
        return result.iloc[0]

    return None