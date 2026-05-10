from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def check_path(label, relative_path, required=True):
    path = PROJECT_ROOT / relative_path
    exists = path.exists()
    status = "OK" if exists else ("MISSING" if required else "OPTIONAL")
    print(f"[{status}] {label}: {path}")
    return exists


def check_import(label, module_name):
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] import {label}: {module_name} ({version})")
        return True
    except Exception as exc:
        print(f"[MISSING] import {label}: {module_name} -> {exc}")
        return False


def check_torch_numpy_bridge():
    try:
        import torch
        import numpy as np

        torch_major = int(torch.__version__.split(".")[0])
        numpy_major = int(np.__version__.split(".")[0])
        if torch_major < 2 and numpy_major >= 2:
            print("[WARN] torch->numpy bridge is likely unavailable -> torch<2 with numpy>=2")
            return False
        print("[OK] torch->numpy bridge version check passed")
        return True
    except Exception as exc:
        print(f"[WARN] torch->numpy bridge is unavailable -> {exc}")
        return False


def main():
    print("Project doctor for the refactored DocRE repository")
    print(f"Workspace: {PROJECT_ROOT}")
    print("")

    print("Dependencies")
    check_import("PyTorch", "torch")
    check_import("NetworkX", "networkx")
    check_import("Joblib", "joblib")
    check_import("NumPy", "numpy")
    check_import("Legacy BERT package", "pytorch_pretrained_bert")
    check_import("Transformers", "transformers")
    check_torch_numpy_bridge()
    print("")

    print("Source layout")
    check_path("Training entry", "run_training.py")
    check_path("Evaluation entry", "run_evaluation.py")
    check_path("Model package", "models")
    check_path("Config package", "config")
    check_path("Legacy duplicate snapshot", "HGR-DREM-master", required=False)
    print("")

    print("Raw inputs")
    check_path("DocRED train_distant", "data/train_distant.json")
    check_path("DocRED train_annotated", "data/train_annotated.json")
    check_path("DocRED dev", "data/dev.json")
    check_path("DocRED test", "data/test.json")
    check_path("BERT weights directory", "bert/bert-base-uncased", required=False)
    check_path("BERT tokenizer vocab", "bert/bert-base-uncased/vocab.txt", required=False)
    check_path("BERT tokenizer json", "bert/bert-base-uncased/tokenizer.json", required=False)
    print("")

    print("Preprocessed assets")
    check_path("GloVe metadata", "prepro_data/rel2id.json")
    check_path("BERT metadata", "new_data/rel2id.json")
    check_path("GloVe vectors", "prepro_data/vec.npy")
    check_path("BERT train shard", "new_data/train0.pkl", required=False)
    check_path("GloVe train shard", "prepro_data/dev_train_3052.pkl", required=False)
    print("")

    print("Recommended commands")
    print("1. python project_doctor.py")
    print("2. python prepare_bert_graph_data.py --in_path ./data --out_path ./new_data")
    print("3. python run_training.py --model_type bert")
    print("4. python run_evaluation.py --model_type bert --save_name bert_docre_mfm_hgr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
