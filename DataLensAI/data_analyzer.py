import re
import pandas as pd


def normalize(text):
    """Make column names and questions easier to compare."""
    text = str(text).lower().replace("_", " ")
    return re.sub(r"[^a-z0-9 ]", " ", text).strip()


def find_id_column(df):
    """Find a column that identifies each record (ID or name)."""
    for column in df.columns:
        name = normalize(column)
        if "id" in name.split() or "name" in name:
            return column
    return None


def find_column(question, df, numeric_only=False):
    """Find the dataset column most relevant to the question."""
    q = normalize(question)
    words = set(q.split())

    columns = df.select_dtypes(
        include="number"
    ).columns if numeric_only else df.columns

    best_column = None
    best_score = 0

    for column in columns:
        name = normalize(column)
        name_words = set(name.split())

        if not name_words:
            continue

        score = len(words & name_words) / len(name_words)
        if f" {name} " in f" {q} ":
            score += 2

        if score > best_score:
            best_score = score
            best_column = column

    if best_score > 0:
        return best_column

    # Use a fallback only when there is one numeric column
    if numeric_only:
        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) == 1:
            return numeric_columns[0]

    return None


def find_group_column(question, df, numeric_column=None):
    """Find a categorical column for group analysis."""
    q = normalize(question)

    preferred_words = [
        "department", "category", "class",
        "course", "branch", "region", "city",
        "team", "group", "gender"
    ]

    candidates = []

    for column in df.columns:
        if column == numeric_column:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        unique_count = df[column].nunique()

        if 2 <= unique_count <= 30:
            candidates.append(column)

    # Prefer a column explicitly mentioned in the question
    for column in candidates:
        if normalize(column) in q:
            return column

    # Otherwise use common category column names
    for word in preferred_words:
        for column in candidates:
            if word in normalize(column):
                return column

    return None


def analyze_question(question, df):
    """
    Perform safe, deterministic analysis using Pandas.

    Returns a dictionary containing the result,
    or None if the question is not supported.
    """
    q = normalize(question)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Dataset overview
    if any(word in q for word in [
        "summarize", "summary", "overview",
        "describe dataset", "dataset information"
    ]):
        missing = int(df.isna().sum().sum())

        return {
            "answer": (
    f"The dataset contains {len(df)} rows "
    f"and {len(df.columns)} columns. "
    f"There are {missing} missing values. "
    f"Numeric columns: "
    f"{', '.join(numeric_columns) if numeric_columns else 'None'}."
),
            "chart_type": None,
            "chart_df": None,
            "chart_title": None
        }

    # Correlation between two numeric columns
    if any(word in q for word in [
        "correlation", "correlated", "relationship"
    ]):
        first = find_column(
            question, df, numeric_only=True
        )

        remaining = [
            col for col in numeric_columns
            if col != first
        ]

        second = None
        best_score = 0
        question_words = set(q.split())

        for col in remaining:
            words = set(normalize(col).split())
            score = len(words & question_words)

            if normalize(col) in q:
                score += 2

            if score > best_score:
                best_score = score
                second = col

        if first and second:
            valid = df[[first, second]].dropna()

            if len(valid) < 2:
                return {
                    "answer": "There are not enough valid rows to calculate correlation.",
                    "chart_type": None,
                    "chart_df": None,
                    "chart_title": None
                }

            correlation = valid[first].corr(valid[second])

            return {
                "answer": (
                    f"The Pearson correlation between "
                    f"{first} and {second} is "
                    f"{correlation:.3f}. "
                    "A positive value indicates a positive "
                    "linear relationship, while a negative "
                    "value indicates a negative one. "
                    "Correlation does not prove causation."
                ),
                "chart_type": "scatter",
                "chart_df": valid,
                "chart_title": f"{first} vs {second}"
            }

        return None

    # Grouped averages
    group_keywords = [
        "by department", "by category", "by class",
        "by course", "by branch", "by region",
        "by city", "by team", "by group",
        "highest average", "lowest average",
        "average for each", "compare departments"
    ]

    if any(word in q for word in group_keywords):
        column = find_column(
            question, df, numeric_only=True
        )

        group_column = find_group_column(
            question, df, column
        )

        if column and group_column:
            grouped = (
                df.groupby(group_column)[column]
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )

            if grouped.empty:
                return None

            if "lowest average" in q:
                category = grouped.idxmin()
                value = grouped.min()
                description = "lowest"
            else:
                category = grouped.idxmax()
                value = grouped.max()
                description = "highest"

            chart_df = (
                grouped.rename("Average")
                .reset_index()
            )

            return {
                "answer": (
                    f"The {description} average "
                    f"{column} is {value:.2f}, "
                    f"recorded by {group_column} "
                    f"'{category}'."
                ),
                "chart_type": "bar",
                "chart_df": chart_df,
                "chart_title": (
                    f"Average {column} by {group_column}"
                )
            }

    # Average or mean
    if any(word in q for word in [
        "average", "mean"
    ]):
        column = find_column(
            question, df, numeric_only=True
        )


        if column:
            value = df[column].mean()

            return {
                "answer": (
                    f"The average {column} is "
                    f"{value:.2f}, calculated from "
                    f"{df[column].count()} non-missing values."
                ),
                "chart_type": None,
                "chart_df": None,
                "chart_title": None
            }

    # Maximum or highest
    if any(word in q for word in [
        "maximum", "highest", "largest",
        "maximum value", "max value"
    ]):
        column = find_column(
            question, df, numeric_only=True
        )

        if column and not df[column].dropna().empty:
            idx = df[column].idxmax()
            value = df.loc[idx, column]
            id_col = find_id_column(df)

            who = (
                f"{id_col} '{df.loc[idx, id_col]}'"
                if id_col else f"row {idx + 1}"
            )

            return {
                "answer": (
                    f"The highest {column} is {value}, "
                    f"recorded by {who}."
                ),
                "chart_type": None,
                "chart_df": None,
                "chart_title": None
            }
            
    # Minimum or lowest
    if any(word in q for word in [
        "minimum", "lowest", "smallest",
        "minimum value", "min value"
    ]):
        column = find_column(
            question, df, numeric_only=True
        )

        if column and not df[column].dropna().empty:
            idx = df[column].idxmin()
            value = df.loc[idx, column]
            id_col = find_id_column(df)

            who = (
                f"{id_col} '{df.loc[idx, id_col]}'"
                if id_col else f"row {idx + 1}"
            )

            return {
                "answer": (
                    f"The lowest {column} is {value}, "
                    f"recorded by {who}."
                ),
                "chart_type": None,
                "chart_df": None,
                "chart_title": None
            }

    # Count rows
    if any(word in q for word in [
        "how many rows", "number of rows",
        "row count", "number of records",
        "how many records"
    ]):
        return {
            "answer": f"The dataset contains {len(df)} rows.",
            "chart_type": None,
            "chart_df": None,
            "chart_title": None
        }

    return None


def build_insights_summary(df):
    """Compute key statistics with Pandas for the AI to explain."""
    lines = [
        f"Rows: {len(df)}, Columns: {len(df.columns)}",
        f"Missing values: {int(df.isna().sum().sum())}"
    ]

    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        lines.append(
            "Numeric summary:\n"
            + numeric.describe().round(2).to_string()
        )

    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        cols = corr.columns
        pairs = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    pairs.append((cols[i], cols[j], value))

        pairs.sort(key=lambda p: abs(p[2]), reverse=True)

        lines.append("Strongest correlations:")
        for a, b, v in pairs[:3]:
            lines.append(f"- {a} vs {b}: {v:.3f}")

    return "\n".join(lines)

def build_insights_summary(df):
    """Compute key statistics with Pandas for the AI to explain."""
    lines = [
        f"Rows: {len(df)}, Columns: {len(df.columns)}",
        f"Missing values: {int(df.isna().sum().sum())}"
    ]

    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        lines.append(
            "Numeric summary:\n"
            + numeric.describe().round(2).to_string()
        )

    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        cols = corr.columns
        pairs = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    pairs.append((cols[i], cols[j], value))

        pairs.sort(key=lambda p: abs(p[2]), reverse=True)

        lines.append("Strongest correlations:")
        for a, b, v in pairs[:3]:
            lines.append(f"- {a} vs {b}: {v:.3f}")

    return "\n".join(lines)

def clean_dataframe(df, remove_duplicates=True, strip_text=True,
                    missing_option="Leave as they are"):
    cleaned = df.copy()
    report = []

    if strip_text:
        for col in cleaned.columns:
            if not pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].apply(
                    lambda v: v.strip() if isinstance(v, str) else v
                )
        report.append("Trimmed extra spaces in text columns")

    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        report.append(
            f"Removed {before - len(cleaned)} duplicate rows"
        )

    missing_before = int(cleaned.isna().sum().sum())

    if missing_option.startswith("Fill"):
        for col in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                if cleaned[col].isna().any():
                    cleaned[col] = cleaned[col].fillna(
                        cleaned[col].median()
                    )
            else:
                cleaned[col] = cleaned[col].fillna("Unknown")

        report.append(f"Filled {missing_before} missing values")

    elif missing_option.startswith("Drop"):
        before = len(cleaned)
        cleaned = cleaned.dropna().reset_index(drop=True)
        report.append(
            f"Dropped {before - len(cleaned)} rows with missing values"
        )

    return cleaned, report