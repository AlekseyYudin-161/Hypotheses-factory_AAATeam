"""Разведка таблицы: python -m src.indexing.inventory data/corpus/файл.xlsx
печатает колонки и первые строки таблицы. Ничего не пишет в базу.
"""

import sys
from src.indexing.parse_tables import read_table

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("укажи путь: python -m src.indexing.inventory <файл>")
        sys.exit(1)

    df = read_table(sys.argv[1])
    print("файл:", sys.argv[1])
    print("строк:", len(df))
    print("колонки:", list(df.columns))
    print("\nпервые 3 строки:\n", df.head(3).to_string())
