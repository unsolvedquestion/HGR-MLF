"""CLI wrapper for GloVe graph-data preprocessing."""

import runpy


def main():
    runpy.run_module("gen_data_extend_graph", run_name="__main__")


if __name__ == "__main__":
    main()
