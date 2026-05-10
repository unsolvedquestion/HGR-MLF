from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from runtime_utils.bert_compat import load_bert_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
CHAR_LIMIT = 16
DEFAULT_STORAGE_SIZE = 10000
SPLIT_NAMES = ("train_distant", "train_annotated", "dev", "test")
SPLIT_CONFIGS = {
    "train_distant": ("train_distant.json", "train", ""),
    "train_annotated": ("train_annotated.json", "dev", "_train"),
    "dev": ("dev.json", "dev", "_dev"),
    "test": ("test.json", "dev", "_test"),
}
RELATION_TYPE = {
    (2, 2): 13535,
    (4, 2): 3673,
    (1, 2): 3146,
    (5, 4): 2269,
    (4, 1): 1955,
    (4, 3): 1848,
    (5, 1): 1698,
    (4, 4): 1552,
    (5, 3): 1448,
    (5, 2): 1402,
    (5, 5): 1367,
    (4, 5): 1117,
    (1, 1): 860,
    (1, 4): 480,
    (2, 1): 458,
    (2, 4): 391,
    (1, 3): 357,
    (1, 5): 260,
    (2, 5): 212,
    (2, 3): 152,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Build BERT graph-preprocessed DocRED shards.")
    parser.add_argument("--in_path", type=str, default="data", help="Directory containing DocRED json files.")
    parser.add_argument("--out_path", type=str, default="new_data", help="Directory for output metadata and shards.")
    parser.add_argument(
        "--bert_model_dir",
        type=str,
        default="./bert/bert-base-uncased",
        help="Local BERT model/tokenizer directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        choices=("all",) + SPLIT_NAMES,
        help="Dataset splits to preprocess.",
    )
    parser.add_argument(
        "--storage_size",
        type=int,
        default=DEFAULT_STORAGE_SIZE,
        help="Number of documents per pickle shard.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of documents processed per split for smoke tests.",
    )
    return parser


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def resolve_splits(raw_splits):
    if "all" in raw_splits:
        return list(SPLIT_NAMES)
    return list(raw_splits)


def encode_token_chars(token, char2id):
    token_chars = np.zeros((CHAR_LIMIT,))
    for char_index, char in enumerate(str(token)):
        if char_index >= CHAR_LIMIT:
            break
        token_chars[char_index] = char2id.get(char, char2id["UNK"])
    return token_chars


def load_pronouns():
    pronoun_path = PROJECT_ROOT / "pronoun_list.txt"
    with pronoun_path.open("r", encoding="utf-8") as handle:
        return [line.strip().lower() for line in handle if line.strip()]


def ensure_required_metadata(out_path: Path):
    required_files = ("rel2id.json", "word2id.json", "char2id.json", "ner2id.json", "char_vec.npy")
    missing = [file_name for file_name in required_files if not (out_path / file_name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing preprocessing metadata in `{out_path}`: {missing_text}")


def prepare_resources(out_path: Path, bert_model_dir: Path):
    ensure_required_metadata(out_path)
    rel2id = load_json(out_path / "rel2id.json")
    dump_json(out_path / "id2rel.json", {value: key for key, value in rel2id.items()})
    resources = {
        "rel2id": rel2id,
        "word2id": load_json(out_path / "word2id.json"),
        "char2id": load_json(out_path / "char2id.json"),
        "ner2id": load_json(out_path / "ner2id.json"),
        "pronouns": load_pronouns(),
        "tokenizer": load_bert_tokenizer(str(bert_model_dir), do_lower_case=True),
    }
    return resources


def build_document_index(document_tokens, tokenizer, char2id):
    cls_token = getattr(tokenizer, "cls_token", "[CLS]") or "[CLS]"
    sep_token = getattr(tokenizer, "sep_token", "[SEP]") or "[SEP]"
    unk_token = getattr(tokenizer, "unk_token", "[UNK]") or "[UNK]"

    bert_tokens = [cls_token]
    token_spans = []
    document_char = []
    for token in document_tokens:
        token_lower = str(token).lower()
        pieces = tokenizer.tokenize(token_lower)
        if not pieces:
            pieces = [unk_token]
        start = len(bert_tokens)
        bert_tokens.extend(pieces)
        end = len(bert_tokens)
        token_spans.append((start, end))
        for piece in pieces:
            document_char.append(encode_token_chars(piece, char2id))
    bert_tokens.append(sep_token)
    document_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    return bert_tokens, token_spans, document_ids, document_char


def build_graph_and_labels(
    raw_item,
    sentence_offsets,
    document_bert,
    token_spans,
    rel2id,
    ner2id,
    pronouns,
):
    vertex_set = raw_item["vertexSet"]
    article_graph = nx.DiGraph()
    node_sent_dict = {}
    max_exist_sentence_num = 0
    max_sentence_length = 0

    for entity_index, mentions in enumerate(vertex_set):
        sentence_ids = []
        article_graph.add_node(
            entity_index,
            exist_sentence=[],
            exist_sentence_id=[],
            exist_pos=[],
            type=[],
        )
        max_exist_sentence_num = max(max_exist_sentence_num, len(mentions))
        for mention in mentions:
            type_id = ner2id[mention["type"]]
            if type_id not in article_graph.nodes[entity_index]["type"]:
                article_graph.nodes[entity_index]["type"].append(type_id)

            sent_id = int(mention["sent_id"])
            sentence_ids.append(sent_id)
            mention["sent_id"] = sent_id

            sentence_offset = sentence_offsets[sent_id]
            pos1, pos2 = mention["pos"]
            token_start = token_spans[pos1 + sentence_offset][0]
            token_end = token_spans[pos2 + sentence_offset - 1][1]
            mention["pos"] = (token_start, token_end)

            sentence_start = token_spans[sentence_offsets[sent_id]][0]
            sentence_end = token_spans[sentence_offsets[sent_id + 1] - 1][1]
            article_graph.nodes[entity_index]["exist_sentence"].append((sentence_start, sentence_end))
            article_graph.nodes[entity_index]["exist_sentence_id"].append(sent_id)
            article_graph.nodes[entity_index]["exist_pos"].append((token_start, token_end))
            max_sentence_length = max(max_sentence_length, len(raw_item["sents"][sent_id]))

        node_sent_dict[entity_index] = sentence_ids

    article_graph.graph["max_entity_exist_num"] = max_exist_sentence_num
    article_graph.graph["all_sentence_num"] = len(sentence_offsets) - 1

    document_length = len(document_bert)
    document_pos = np.zeros((document_length,))
    document_ner = np.zeros((document_length,))
    for entity_index, mentions in enumerate(vertex_set, 1):
        for mention in mentions:
            document_pos[mention["pos"][0] : mention["pos"][1]] = entity_index
            document_ner[mention["pos"][0] : mention["pos"][1]] = ner2id[mention["type"]]

    labels = raw_item.get("labels", [])
    label_matrix = np.zeros((len(vertex_set), len(vertex_set), len(rel2id)))
    label_evidence = {}
    count_all_label_rel = len(labels)
    for label in labels:
        relation_name = label["r"]
        if relation_name not in rel2id:
            raise KeyError(f"Unknown relation `{relation_name}` in item `{raw_item['title']}`")
        relation_id = rel2id[relation_name]
        label["r"] = relation_id
        label_matrix[label["h"], label["t"], relation_id] = 1
        evidence = np.zeros(len(sentence_offsets) - 1)
        for sentence_id in label["evidence"]:
            evidence[sentence_id] = 1
        label_evidence[(label["h"], label["t"], relation_id)] = evidence

    for head_index in range(len(vertex_set)):
        for tail_index in range(len(vertex_set)):
            if label_matrix[head_index, tail_index, :].sum() == 0:
                label_matrix[head_index, tail_index, 0] = 1

    label_mask = []
    max_coexist_sentence_num = 0
    max_coexist_sentence_length = 0
    for node_h in article_graph.nodes():
        for node_t in article_graph.nodes():
            if node_h == node_t:
                continue

            common_sentences = []
            common_positions = []
            common_next_sentences = []
            common_next_positions = []
            sentences_h = article_graph.nodes[node_h]["exist_sentence"]
            positions_h = article_graph.nodes[node_h]["exist_pos"]
            sentences_t = article_graph.nodes[node_t]["exist_sentence"]
            positions_t = article_graph.nodes[node_t]["exist_pos"]

            for pos_h, sent_h in zip(positions_h, sentences_h):
                for pos_t, sent_t in zip(positions_t, sentences_t):
                    if sent_h == sent_t:
                        common_sentences.append(sent_h)
                        common_positions.append(pos_h + pos_t)
                        max_coexist_sentence_length = max(max_coexist_sentence_length, sent_h[1] - sent_h[0])

                    if sent_h[1] == sent_t[0]:
                        if any(token.lower() in pronouns for token in document_bert[sent_h[0] : sent_t[1]]):
                            common_next_sentences.append((sent_h[0], sent_t[1]))
                            common_next_positions.append(pos_h + pos_t)
                    if sent_h[0] == sent_t[1]:
                        if any(token.lower() in pronouns for token in document_bert[sent_t[0] : sent_h[1]]):
                            common_next_sentences.append((sent_t[0], sent_h[1]))
                            common_next_positions.append(pos_h + pos_t)

            if common_sentences:
                article_graph.add_edge(node_h, node_t, sentences=common_sentences, position=common_positions)
                max_coexist_sentence_num = max(max_coexist_sentence_num, len(common_sentences))
            elif common_next_sentences:
                article_graph.add_edge(
                    node_h,
                    node_t,
                    sentences=common_next_sentences,
                    position=common_next_positions,
                )
                max_coexist_sentence_num = max(max_coexist_sentence_num, len(common_next_sentences))

            entity_types_h = article_graph.nodes[node_h]["type"]
            entity_types_t = article_graph.nodes[node_t]["type"]
            if any((type_h, type_t) in RELATION_TYPE for type_h in entity_types_h for type_t in entity_types_t):
                label_mask.append((node_h, node_t))

    article_graph.graph["max_sentence_length"] = max_coexist_sentence_length
    article_graph.graph["max_sentence_num"] = max_coexist_sentence_num

    return {
        "graph": article_graph,
        "vertexSet": vertex_set,
        "node_sent_dict": node_sent_dict,
        "document_pos": document_pos,
        "document_ner": document_ner,
        "labels": labels,
        "label_matrix": label_matrix,
        "label_evidence": label_evidence,
        "label_mask": label_mask,
        "count_all_label_rel": count_all_label_rel,
        "max_exist_sentence_num": max_exist_sentence_num,
        "max_sentence_length": max_sentence_length,
        "max_coexist_sentence_num": max_coexist_sentence_num,
        "max_coexist_sentence_length": max_coexist_sentence_length,
    }


def write_shard(out_path: Path, name_prefix: str, suffix: str, shard_index: int, data):
    output_path = out_path / f"{name_prefix}{suffix}{shard_index}.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_split(
    split_name: str,
    input_path: Path,
    out_path: Path,
    storage_size: int,
    limit: int | None,
    resources,
):
    file_name, name_prefix, suffix = SPLIT_CONFIGS[split_name]
    dataset = load_json(input_path / file_name)
    if limit is not None:
        dataset = dataset[:limit]
    if not dataset:
        raise ValueError(f"Split `{split_name}` has no documents to preprocess.")

    tokenizer = resources["tokenizer"]
    char2id = resources["char2id"]
    rel2id = resources["rel2id"]
    ner2id = resources["ner2id"]
    pronouns = resources["pronouns"]

    max_exist_sentence_num = 0
    max_sentence_length = 0
    max_coexist_sentence_num = 0
    max_coexist_sentence_length = 0
    count_all_label_rel = 0

    buffered_items = []
    shard_index = 0
    for raw_item in tqdm(dataset, desc=f"{split_name}"):
        sentence_offsets = [0]
        flat_document = []
        total_length = 0
        for sentence in raw_item["sents"]:
            flat_document.extend(sentence)
            total_length += len(sentence)
            sentence_offsets.append(total_length)

        document_bert, token_spans, document_ids, document_char = build_document_index(
            flat_document,
            tokenizer,
            char2id,
        )
        graph_payload = build_graph_and_labels(
            raw_item,
            sentence_offsets,
            document_bert,
            token_spans,
            rel2id,
            ner2id,
            pronouns,
        )

        title = raw_item["title"]
        buffered_items.append(
            {
                "document": document_ids,
                "document_char": document_char,
                "title": title,
                "title_char": [encode_token_chars(char.lower(), char2id) for char in str(title)],
                "graph": graph_payload["graph"],
                "vertexSet": graph_payload["vertexSet"],
                "node_sent_dict": graph_payload["node_sent_dict"],
                "document_pos": graph_payload["document_pos"],
                "document_ner": graph_payload["document_ner"],
                "labels": graph_payload["labels"],
                "label_matrix": graph_payload["label_matrix"],
                "label_evidence": graph_payload["label_evidence"],
                "label_mask": graph_payload["label_mask"],
                "Ls": sentence_offsets,
            }
        )

        count_all_label_rel += graph_payload["count_all_label_rel"]
        max_exist_sentence_num = max(max_exist_sentence_num, graph_payload["max_exist_sentence_num"])
        max_sentence_length = max(max_sentence_length, graph_payload["max_sentence_length"])
        max_coexist_sentence_num = max(max_coexist_sentence_num, graph_payload["max_coexist_sentence_num"])
        max_coexist_sentence_length = max(
            max_coexist_sentence_length,
            graph_payload["max_coexist_sentence_length"],
        )

        if len(buffered_items) >= storage_size:
            write_shard(out_path, name_prefix, suffix, shard_index, buffered_items)
            buffered_items = []
            shard_index += 1

    if buffered_items:
        write_shard(out_path, name_prefix, suffix, shard_index, buffered_items)

    print(f"[{split_name}] count_all_label_rel: {count_all_label_rel}")
    print(f"[{split_name}] data_len: {len(dataset)}")
    print(f"[{split_name}] max_entity_mentions: {max_exist_sentence_num}")
    print(f"[{split_name}] max_sentence_length: {max_sentence_length}")
    print(f"[{split_name}] max_pair_sentence_count: {max_coexist_sentence_num}")
    print(f"[{split_name}] max_pair_sentence_length: {max_coexist_sentence_length}")


def main():
    args = build_parser().parse_args()

    input_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    bert_model_dir = Path(args.bert_model_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    resources = prepare_resources(out_path, bert_model_dir)
    for split_name in resolve_splits(args.splits):
        process_split(
            split_name=split_name,
            input_path=input_path,
            out_path=out_path,
            storage_size=args.storage_size,
            limit=args.limit,
            resources=resources,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
