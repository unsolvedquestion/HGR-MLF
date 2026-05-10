from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
CHAR_LIMIT = 16
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
    parser = argparse.ArgumentParser(description="Build GloVe graph-preprocessed DocRED files.")
    parser.add_argument("--in_path", type=str, default="./data", help="Directory containing DocRED json files.")
    parser.add_argument("--out_path", type=str, default="./prepro_data", help="Directory for output metadata and files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        choices=("all",) + SPLIT_NAMES,
        help="Dataset splits to preprocess.",
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


def load_pronouns():
    pronoun_path = PROJECT_ROOT / "pronoun_list.txt"
    with pronoun_path.open("r", encoding="utf-8") as handle:
        return [line.strip().lower() for line in handle if line.strip()]


def ensure_required_metadata(out_path: Path):
    required_files = ("rel2id.json", "word2id.json", "char2id.json", "ner2id.json")
    missing = [file_name for file_name in required_files if not (out_path / file_name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing preprocessing metadata in `{out_path}`: {missing_text}")


def prepare_resources(out_path: Path):
    ensure_required_metadata(out_path)
    rel2id = load_json(out_path / "rel2id.json")
    dump_json(out_path / "id2rel.json", {value: key for key, value in rel2id.items()})
    resources = {
        "rel2id": rel2id,
        "word2id": load_json(out_path / "word2id.json"),
        "char2id": load_json(out_path / "char2id.json"),
        "ner2id": load_json(out_path / "ner2id.json"),
        "pronouns": load_pronouns(),
    }
    return resources


def encode_token_chars(token, char2id):
    token_chars = np.zeros((CHAR_LIMIT,))
    for char_index, char in enumerate(str(token)):
        if char_index >= CHAR_LIMIT:
            break
        token_chars[char_index] = char2id.get(char, char2id["UNK"])
    return token_chars


def vectorize_document(raw_item, word2id, char2id):
    sentence_offsets = [0]
    flat_document = []
    total_length = 0
    for sentence in raw_item["sents"]:
        flat_document.extend(sentence)
        total_length += len(sentence)
        sentence_offsets.append(total_length)

    document_ids = []
    document_char = []
    for token in flat_document:
        token_lower = str(token).lower()
        document_ids.append(word2id.get(token_lower, word2id["UNK"]))
        document_char.append(encode_token_chars(token_lower, char2id))
    return flat_document, sentence_offsets, document_ids, document_char


def build_graph_and_labels(raw_item, sentence_offsets, document_tokens, rel2id, ner2id, pronouns):
    vertex_set = raw_item["vertexSet"]
    article_graph = nx.DiGraph()
    max_exist_sentence_num = 0
    max_sentence_length = 0

    for entity_index, mentions in enumerate(vertex_set):
        article_graph.add_node(entity_index, exist_sentence=[], exist_pos=[], type=[])
        max_exist_sentence_num = max(max_exist_sentence_num, len(mentions))
        for mention in mentions:
            type_id = ner2id[mention["type"]]
            if type_id not in article_graph.nodes[entity_index]["type"]:
                article_graph.nodes[entity_index]["type"].append(type_id)

            sent_id = int(mention["sent_id"])
            mention["sent_id"] = sent_id
            sentence_offset = sentence_offsets[sent_id]
            pos1, pos2 = mention["pos"]
            mention["pos"] = (pos1 + sentence_offset, pos2 + sentence_offset)

            article_graph.nodes[entity_index]["exist_sentence"].append(
                (sentence_offsets[sent_id], sentence_offsets[sent_id + 1])
            )
            article_graph.nodes[entity_index]["exist_pos"].append(mention["pos"])
            max_sentence_length = max(max_sentence_length, len(raw_item["sents"][sent_id]))

    article_graph.graph["max_entity_exist_num"] = max_exist_sentence_num

    document_length = len(document_tokens)
    document_pos = np.zeros((document_length,))
    document_ner = np.zeros((document_length,))
    for entity_index, mentions in enumerate(vertex_set, 1):
        for mention in mentions:
            document_pos[mention["pos"][0] : mention["pos"][1]] = entity_index
            document_ner[mention["pos"][0] : mention["pos"][1]] = ner2id[mention["type"]]

    labels = raw_item.get("labels", [])
    label_matrix = np.zeros((len(vertex_set), len(vertex_set), len(rel2id)))
    for label in labels:
        relation_name = label["r"]
        if relation_name not in rel2id:
            raise KeyError(f"Unknown relation `{relation_name}` in item `{raw_item['title']}`")
        relation_id = rel2id[relation_name]
        label["r"] = relation_id
        label_matrix[label["h"], label["t"], relation_id] = 1

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
                        if any(token.lower() in pronouns for token in document_tokens[sent_h[0] : sent_t[1]]):
                            common_next_sentences.append((sent_h[0], sent_t[1]))
                            common_next_positions.append(pos_h + pos_t)
                    if sent_h[0] == sent_t[1]:
                        if any(token.lower() in pronouns for token in document_tokens[sent_t[0] : sent_h[1]]):
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
        "document_pos": document_pos,
        "document_ner": document_ner,
        "labels": labels,
        "label_matrix": label_matrix,
        "label_mask": label_mask,
        "max_exist_sentence_num": max_exist_sentence_num,
        "max_sentence_length": max_sentence_length,
        "max_coexist_sentence_num": max_coexist_sentence_num,
        "max_coexist_sentence_length": max_coexist_sentence_length,
    }


def process_split(split_name: str, input_path: Path, out_path: Path, limit: int | None, resources):
    file_name, name_prefix, suffix = SPLIT_CONFIGS[split_name]
    dataset = load_json(input_path / file_name)
    if limit is not None:
        dataset = dataset[:limit]
    if not dataset:
        raise ValueError(f"Split `{split_name}` has no documents to preprocess.")

    word2id = resources["word2id"]
    char2id = resources["char2id"]
    rel2id = resources["rel2id"]
    ner2id = resources["ner2id"]
    pronouns = resources["pronouns"]

    data = []
    max_exist_sentence_num = 0
    max_sentence_length = 0
    max_coexist_sentence_num = 0
    max_coexist_sentence_length = 0

    for raw_item in tqdm(dataset, desc=f"{split_name}"):
        document_tokens, sentence_offsets, document_ids, document_char = vectorize_document(raw_item, word2id, char2id)
        graph_payload = build_graph_and_labels(
            raw_item,
            sentence_offsets,
            document_tokens,
            rel2id,
            ner2id,
            pronouns,
        )

        title = raw_item["title"]
        data.append(
            {
                "document": document_ids,
                "document_char": document_char,
                "title": title,
                "title_char": [encode_token_chars(char.lower(), char2id) for char in str(title)],
                "vertexSet": graph_payload["vertexSet"],
                "document_pos": graph_payload["document_pos"],
                "document_ner": graph_payload["document_ner"],
                "labels": graph_payload["labels"],
                "label_matrix": graph_payload["label_matrix"],
                "graph": graph_payload["graph"],
                "label_mask": graph_payload["label_mask"],
                "Ls": sentence_offsets,
            }
        )

        max_exist_sentence_num = max(max_exist_sentence_num, graph_payload["max_exist_sentence_num"])
        max_sentence_length = max(max_sentence_length, graph_payload["max_sentence_length"])
        max_coexist_sentence_num = max(max_coexist_sentence_num, graph_payload["max_coexist_sentence_num"])
        max_coexist_sentence_length = max(
            max_coexist_sentence_length,
            graph_payload["max_coexist_sentence_length"],
        )

    output_name = f"{name_prefix}{suffix}_{len(data) - 1}.pkl"
    joblib.dump(data, out_path / output_name)

    print(f"[{split_name}] data_len: {len(dataset)}")
    print(f"[{split_name}] max_entity_mentions: {max_exist_sentence_num}")
    print(f"[{split_name}] max_sentence_length: {max_sentence_length}")
    print(f"[{split_name}] max_pair_sentence_count: {max_coexist_sentence_num}")
    print(f"[{split_name}] max_pair_sentence_length: {max_coexist_sentence_length}")


def main():
    args = build_parser().parse_args()

    input_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    resources = prepare_resources(out_path)
    for split_name in resolve_splits(args.splits):
        process_split(
            split_name=split_name,
            input_path=input_path,
            out_path=out_path,
            limit=args.limit,
            resources=resources,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
