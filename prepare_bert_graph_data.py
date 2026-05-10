"""CLI wrapper for BERT graph-data preprocessing."""

import runpy


def main():
    runpy.run_module("gen_bert_data_extend_graph", run_name="__main__")


if __name__ == "__main__":
    main()
