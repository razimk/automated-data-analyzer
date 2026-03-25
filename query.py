def simple_nl_query(df, query):

    query = query.lower()

    # find column mentioned in query
    column_found = None

    for col in df.columns:
        if col.lower() in query:
            column_found = col
            break

    # Average
    if "average" in query or "mean" in query:
        if column_found:
            return f"Average {column_found}: {df[column_found].mean()}"
        else:
            return df.mean(numeric_only=True)

    # Sum
    elif "sum" in query or "total" in query:
        if column_found:
            return f"Total {column_found}: {df[column_found].sum()}"
        else:
            return df.sum(numeric_only=True)

    # Maximum
    elif "max" in query:
        if column_found:
            return f"Max {column_found}: {df[column_found].max()}"
        else:
            return df.max(numeric_only=True)

    # Minimum
    elif "min" in query:
        if column_found:
            return f"Min {column_found}: {df[column_found].min()}"
        else:
            return df.min(numeric_only=True)

    else:
        return "Query not understood. Try: average income, total sales, max age"