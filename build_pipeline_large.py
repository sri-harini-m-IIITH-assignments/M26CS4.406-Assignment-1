import os
from python_scripts_large.make_predictions import generate_prediction_file, zip_prediction
from python_scripts_large.question1_large import run_q1_mind, run_q1_ebnerd
from python_scripts_large.question2_large import run_q2
from python_scripts_large.question3_large import run_q3
from python_scripts_large.question4_large import run_q4

def main():
    run_q1_mind()
    run_q1_ebnerd()

    run_q2()
    run_q3()
    run_q4()

    configs = [
        {
            "dataset_type": "mind",
            "test_path_or_dir": "data_large/MINDlarge_test/behaviors.tsv",
            "method": "bm25",
            "bm25_index_path": "bm25_indexes_large/mind_bm25.pkl",
            "faiss_index_path": "faiss_indexes_large/mind_faiss.pkl",
            "language": "english",
            "output_dir": "predictions_output/mind_bm25",
        },
        {
            "dataset_type": "mind",
            "test_path_or_dir": "data_large/MINDlarge_test/behaviors.tsv",
            "method": "faiss",
            "bm25_index_path": "bm25_indexes_large/mind_bm25.pkl",
            "faiss_index_path": "faiss_indexes_large/mind_faiss.pkl",
            "language": "english",
            "output_dir": "predictions_output/mind_faiss",
        },
        {
            "dataset_type": "ebnerd",
            "test_path_or_dir": "data_large/ebnerd_large/test",  
            "method": "bm25",
            "bm25_index_path": "bm25_indexes_large/ebnerd_large_bm25.pkl",
            "faiss_index_path": "faiss_indexes_large/ebnerd_large_faiss.pkl",
            "language": "danish",
            "output_dir": "predictions_output/ebnerd_bm25",
        },
        {
            "dataset_type": "ebnerd",
            "test_path_or_dir": "data_large/ebnerd_large/test", 
            "method": "faiss",
            "bm25_index_path": "bm25_indexes_large/ebnerd_large_bm25.pkl",
            "faiss_index_path": "faiss_indexes_large/ebnerd_large_faiss.pkl",
            "language": "danish",
            "output_dir": "predictions_output/ebnerd_faiss",
        },
    ]

    for cfg in configs:
        print(f"\n--- Generating predictions for [{cfg['dataset_type'].upper()} | {cfg['method'].upper()}] ---")
        pred_path = generate_prediction_file(
            dataset_type=cfg["dataset_type"],
            test_path_or_dir=cfg["test_path_or_dir"],
            method=cfg["method"],
            bm25_index_path=cfg["bm25_index_path"],
            faiss_index_path=cfg["faiss_index_path"],
            language=cfg["language"],
            output_dir=cfg["output_dir"],
        )
        zip_prediction(pred_path, dataset_type=cfg["dataset_type"])

if __name__ == "__main__":
    main()