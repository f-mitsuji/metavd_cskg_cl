from pathlib import Path

import pandas as pd

from src.settings import AUTO_METAVD_DIR, METAVD_DIR

VECTORIZATIONS = ["numberbatch", "sent2vec", "word2vec", "mpnet"]
THRESHOLDS_DEFAULT = [0.8, 0.85]
THRESHOLDS_MPNET = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9]


def normalize_relation_pair(dataset1, idx1, dataset2, idx2):
    if (dataset1, idx1) <= (dataset2, idx2):
        return (dataset1, idx1, dataset2, idx2)
    else:
        return (dataset2, idx2, dataset1, idx1)


def create_relation_pairs_normalized(df):
    pairs = set()
    for _, row in df.iterrows():
        normalized_pair = normalize_relation_pair(
            row["from_dataset"], int(row["from_action_idx"]), row["to_dataset"], int(row["to_action_idx"])
        )
        pairs.add(normalized_pair)
    return pairs


def load_llm_evaluation_results(llm_results_path):
    if not Path(llm_results_path).exists():
        return None

    llm_df = pd.read_csv(llm_results_path)

    llm_results = {}
    for _, row in llm_df.iterrows():
        normalized_pair = normalize_relation_pair(
            row["dataset1"], int(row["action1_idx"]), row["dataset2"], int(row["action2_idx"])
        )
        llm_results[normalized_pair] = row["llm_is_equal"]

    return llm_results


def calculate_metrics(
    predicted_file,
    ground_truth_file,
    llm_results_file=None,
    verbose=False,
):
    try:
        pred_df = pd.read_csv(predicted_file)
    except Exception as e:
        if verbose:
            print(f"Error loading predicted file {predicted_file}: {e}")
        return None

    try:
        gt_df = pd.read_csv(ground_truth_file)
        gt_equal_df = gt_df[gt_df["relation"] == "equal"].copy()
    except Exception as e:
        if verbose:
            print(f"Error loading ground truth file {ground_truth_file}: {e}")
        return None

    predicted_pairs = create_relation_pairs_normalized(pred_df)
    gt_pairs = create_relation_pairs_normalized(gt_equal_df)

    # TP, FP, FN
    true_positives = predicted_pairs & gt_pairs
    false_positives = predicted_pairs - gt_pairs
    false_negatives = gt_pairs - predicted_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # LLM評価結果の適用
    llm_metrics = None
    if llm_results_file and Path(llm_results_file).exists():
        llm_results = load_llm_evaluation_results(llm_results_file)

        if llm_results:
            llm_identified_tp = 0
            not_in_llm_results = 0

            for fp_pair in false_positives:
                if fp_pair in llm_results:
                    if llm_results[fp_pair]:  # llm_is_equal == True
                        llm_identified_tp += 1
                else:
                    not_in_llm_results += 1

            # 調整後のメトリクス
            adjusted_tp = tp + llm_identified_tp
            adjusted_fp = fp - llm_identified_tp
            adjusted_fn = fn

            adjusted_precision = adjusted_tp / (adjusted_tp + adjusted_fp) if (adjusted_tp + adjusted_fp) > 0 else 0.0
            adjusted_recall = adjusted_tp / (adjusted_tp + adjusted_fn) if (adjusted_tp + adjusted_fn) > 0 else 0.0
            adjusted_f1 = (
                2 * (adjusted_precision * adjusted_recall) / (adjusted_precision + adjusted_recall)
                if (adjusted_precision + adjusted_recall) > 0
                else 0.0
            )

            llm_metrics = {
                "precision": adjusted_precision,
                "recall": adjusted_recall,
                "f1_score": adjusted_f1,
                "true_positives": adjusted_tp,
                "false_positives": adjusted_fp,
                "false_negatives": adjusted_fn,
                "llm_identified_tp": llm_identified_tp,
            }

    return {
        "normalized": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "llm_adjusted": llm_metrics,
    }


def run_all_evaluations():
    ground_truth_file = METAVD_DIR / "metavd_v1.csv"

    if not ground_truth_file.exists():
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return

    all_results = []

    print(f"\n{'='*80}")
    print("BATCH EVALUATION")
    print(f"{'='*80}")
    print(f"Ground truth file: {ground_truth_file}\n")

    for vectorization in VECTORIZATIONS:
        thresholds = THRESHOLDS_MPNET if vectorization == "mpnet" else THRESHOLDS_DEFAULT

        for threshold in thresholds:
            predicted_file = AUTO_METAVD_DIR / f"auto_metavd_{vectorization}_{threshold}.csv"
            llm_results_file = AUTO_METAVD_DIR / f"llm_evaluation_results_{vectorization}_{threshold}.csv"

            if not predicted_file.exists():
                print(f"[SKIP] {vectorization} @ {threshold}: Predicted file not found")
                continue

            metrics = calculate_metrics(
                predicted_file,
                ground_truth_file,
                llm_results_file=llm_results_file,
            )

            if metrics:
                result = {
                    "vectorization": vectorization,
                    "threshold": threshold,
                    "precision": metrics["normalized"]["precision"],
                    "recall": metrics["normalized"]["recall"],
                    "f1_score": metrics["normalized"]["f1_score"],
                    "tp": metrics["normalized"]["true_positives"],
                    "fp": metrics["normalized"]["false_positives"],
                    "fn": metrics["normalized"]["false_negatives"],
                }

                if metrics["llm_adjusted"]:
                    result["llm_precision"] = metrics["llm_adjusted"]["precision"]
                    result["llm_recall"] = metrics["llm_adjusted"]["recall"]
                    result["llm_f1_score"] = metrics["llm_adjusted"]["f1_score"]
                    result["llm_tp"] = metrics["llm_adjusted"]["true_positives"]
                    result["llm_fp"] = metrics["llm_adjusted"]["false_positives"]
                    result["llm_identified_tp"] = metrics["llm_adjusted"]["llm_identified_tp"]
                else:
                    result["llm_precision"] = None
                    result["llm_recall"] = None
                    result["llm_f1_score"] = None
                    result["llm_tp"] = None
                    result["llm_fp"] = None
                    result["llm_identified_tp"] = None

                all_results.append(result)

                llm_info = ""
                if metrics["llm_adjusted"]:
                    llm_info = f" | LLM: P={result['llm_precision']:.4f}, R={result['llm_recall']:.4f}, F1={result['llm_f1_score']:.4f}"
                print(
                    f"[OK] {vectorization:12} @ {threshold:.2f}: "
                    f"P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1_score']:.4f} "
                    f"(TP={result['tp']}, FP={result['fp']}, FN={result['fn']}){llm_info}"
                )

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = AUTO_METAVD_DIR / "evaluation_results_all.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {results_csv}")

        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(
            f"{'Vectorization':<12} | {'Threshold':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'LLM F1':<9}"
        )
        print("-" * 80)
        for r in all_results:
            llm_f1_str = f"{r['llm_f1_score']:.4f}" if r["llm_f1_score"] is not None else "N/A"
            print(
                f"{r['vectorization']:<12} | {r['threshold']:<9.2f} | {r['precision']:<9.4f} | "
                f"{r['recall']:<9.4f} | {r['f1_score']:<9.4f} | {llm_f1_str:<9}"
            )

        results_txt = AUTO_METAVD_DIR / "evaluation_results_all.txt"
        with results_txt.open("w", encoding="utf-8") as f:
            f.write("Action Relation Extraction Evaluation Results\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"{'Vectorization':<12} | {'Threshold':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'LLM F1':<9}\n"
            )
            f.write("-" * 80 + "\n")
            for r in all_results:
                llm_f1_str = f"{r['llm_f1_score']:.4f}" if r["llm_f1_score"] is not None else "N/A"
                f.write(
                    f"{r['vectorization']:<12} | {r['threshold']:<9.2f} | {r['precision']:<9.4f} | "
                    f"{r['recall']:<9.4f} | {r['f1_score']:<9.4f} | {llm_f1_str:<9}\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for r in all_results:
                f.write(f"{r['vectorization']} @ {r['threshold']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Precision: {r['precision']:.4f}\n")
                f.write(f"  Recall: {r['recall']:.4f}\n")
                f.write(f"  F1-Score: {r['f1_score']:.4f}\n")
                f.write(f"  TP: {r['tp']}, FP: {r['fp']}, FN: {r['fn']}\n")
                if r["llm_f1_score"] is not None:
                    f.write("  [LLM Adjusted]\n")
                    f.write(f"  LLM Precision: {r['llm_precision']:.4f}\n")
                    f.write(f"  LLM Recall: {r['llm_recall']:.4f}\n")
                    f.write(f"  LLM F1-Score: {r['llm_f1_score']:.4f}\n")
                    f.write(f"  LLM identified TP: {r['llm_identified_tp']}\n")
                f.write("\n")

        print(f"Results also saved to: {results_txt}")


def main():
    run_all_evaluations()


if __name__ == "__main__":
    main()
