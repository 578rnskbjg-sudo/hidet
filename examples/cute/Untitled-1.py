

def markdown_table_to_dicts(markdown_file):
    """
    Converts a Markdown table string into a list of dictionaries.
    Each dictionary represents a row, with keys being the column headers.
    """
    with open(markdown_file, "r") as f:
        markdown_table_string = f.read()
    lines = markdown_table_string.strip().split('\n')
    if len(lines) < 2:
        return []  # Not a valid table (needs at least header and separator)

    # Extract headers
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]

    # Skip the separator line (lines[1])
    data_rows = lines[2:]
    header_values = [[] for _ in headers]

    for row_str in data_rows:
        values = [v.strip() for v in row_str.split('|') if v.strip()]
        if len(values) == len(headers):
            for value, header_value in zip(values, header_values):
                header_value.append(value)
    return header_values


def main():
        