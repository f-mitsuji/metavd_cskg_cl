import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.settings import AUTO_METAVD_DIR, METAVD_DIR

# VECTORIZATION = "numberbatch"
# VECTORIZATION = "sent2vec"
# VECTORIZATION = "word2vec"
VECTORIZATION = "mpnet"
# THRESHOLD = 0.8
THRESHOLD = 0.81
# THRESHOLD = 0.82
# THRESHOLD = 0.83
# THRESHOLD = 0.84
# THRESHOLD = 0.85
# THRESHOLD = 0.86
# THRESHOLD = 0.87
# THRESHOLD = 0.88
# THRESHOLD = 0.89
# THRESHOLD = 0.9


def normalize_relation_pair(dataset1, idx1, dataset2, idx2):
    if (dataset1, idx1) <= (dataset2, idx2):
        return (dataset1, idx1, dataset2, idx2)
    else:
        return (dataset2, idx2, dataset1, idx1)


def create_relation_pairs_original(df):
    pairs = set()
    for _, row in df.iterrows():
        pair = (row["from_dataset"], int(row["from_action_idx"]), row["to_dataset"], int(row["to_action_idx"]))
        pairs.add(pair)
    return pairs


def create_relation_pairs_normalized(df):
    pairs = set()
    for _, row in df.iterrows():
        normalized_pair = normalize_relation_pair(
            row["from_dataset"], int(row["from_action_idx"]), row["to_dataset"], int(row["to_action_idx"])
        )
        pairs.add(normalized_pair)
    return pairs


def evaluate_relation_with_llm(action1_name, action2_name, client, model="gpt-5-mini"):
    prompt = f"""You are evaluating whether two action labels can be treated as the same label for training an action recognition model.

Context: We are building a training dataset for an action recognition model by combining multiple action recognition datasets. When training the model, videos with the same action label will be used together as training examples for that action class.

Question: Can videos labeled as "{action1_name}" and videos labeled as "{action2_name}" be used together as the same action class for training?

Consider:
- Do these labels represent essentially the same physical action or activity?
- Would using videos from both labels together help or hurt the model's ability to learn this action?
- Minor variations (synonyms, tense, spelling) are acceptable if the underlying action is the same
- Different actions should remain separate even if related

Respond in JSON format with only:
{{
    "is_equal": true or false
}}
"""

    try:
        response = client.responses.create(model=model, input=prompt)

        output_text = response.output_text.strip()
        return json.loads(output_text)

    except Exception as e:  # noqa: BLE001
        print(f"Error in LLM evaluation: {e}")
        return {"is_equal": False}


def llm_evaluate_false_positives(
    pred_df, gt_df, false_positives_norm, output_folder, confidence_threshold=0.7, max_evaluations=None
):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return None

    client = OpenAI(api_key=api_key)

    print(f"\n{'='*60}")
    print("LLM EVALUATION OF FALSE POSITIVES")
    print(f"{'='*60}")
    print(f"Total False Positives to evaluate: {len(false_positives_norm)}")

    if max_evaluations:
        fp_list = list(false_positives_norm)[:max_evaluations]
        print(f"Evaluating first {max_evaluations} False Positives")
    else:
        fp_list = list(false_positives_norm)
        print(f"Evaluating all {len(fp_list)} False Positives")

    pred_action_map = {}
    for _, row in pred_df.iterrows():
        key = (row["from_dataset"], int(row["from_action_idx"]), row["to_dataset"], int(row["to_action_idx"]))
        pred_action_map[key] = (row["from_action_name"], row["to_action_name"])

    llm_results = []
    llm_true_positives = []

    for i, fp_norm in enumerate(fp_list, 1):
        ds1, idx1, ds2, idx2 = fp_norm

        action1_name = None
        action2_name = None

        key1 = (ds1, idx1, ds2, idx2)
        key2 = (ds2, idx2, ds1, idx1)

        if key1 in pred_action_map:
            action1_name, action2_name = pred_action_map[key1]
        elif key2 in pred_action_map:
            action2_name, action1_name = pred_action_map[key2]

        if not action1_name or not action2_name:
            print(f"Warning: Could not find action names for pair {fp_norm}")
            continue

        print(f"\n[{i}/{len(fp_list)}] Evaluating: {ds1}[{idx1}]:'{action1_name}' <-> {ds2}[{idx2}]:'{action2_name}'")

        llm_result = evaluate_relation_with_llm(action1_name, action2_name, client)

        result_record = {
            "dataset1": ds1,
            "action1_idx": idx1,
            "action1_name": action1_name,
            "dataset2": ds2,
            "action2_idx": idx2,
            "action2_name": action2_name,
            "llm_is_equal": llm_result["is_equal"],
        }

        llm_results.append(result_record)

        print(f"  LLM Result: is_equal={llm_result['is_equal']}")

        if llm_result["is_equal"]:
            llm_true_positives.append(fp_norm)
            print("  ✓ Identified as potential True Positive!")

        time.sleep(0.5)

    llm_results_df = pd.DataFrame(llm_results)
    llm_results_path = Path(output_folder) / f"llm_evaluation_results_{VECTORIZATION}_{THRESHOLD}.csv"
    llm_results_df.to_csv(llm_results_path, index=False)
    print(f"\nLLM evaluation results saved to: {llm_results_path}")

    total_evaluated = len(llm_results)
    identified_as_equal = sum(1 for r in llm_results if r["llm_is_equal"])
    high_confidence_equal = len(llm_true_positives)

    print(f"\n{'='*60}")
    print("LLM EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total evaluated: {total_evaluated}")
    print(f"Identified as equal: {identified_as_equal} ({identified_as_equal/total_evaluated*100:.1f}%)")

    return {
        "total_evaluated": total_evaluated,
        "identified_as_equal": identified_as_equal,
        "high_confidence_equal": high_confidence_equal,
        "llm_true_positives": llm_true_positives,
        "llm_results_df": llm_results_df,
    }


def calculate_metrics(
    predicted_file,
    ground_truth_file,
    output_folder=None,
    enable_llm_evaluation=False,
    llm_confidence_threshold=0.7,
    llm_max_evaluations=None,
):
    try:
        pred_df = pd.read_csv(predicted_file)
        print(f"Loaded predicted relations: {len(pred_df)}")
    except Exception as e:
        print(f"Error loading predicted file {predicted_file}: {e}")
        return None

    try:
        gt_df = pd.read_csv(ground_truth_file)
        gt_equal_df = gt_df[gt_df["relation"] == "equal"].copy()
        print(f"Loaded ground truth relations (total): {len(gt_df)}")
        print(f"Ground truth equal relations: {len(gt_equal_df)}")
    except Exception as e:
        print(f"Error loading ground truth file {ground_truth_file}: {e}")
        return None

    print(f"\n{'='*60}")
    print("ORIGINAL EVALUATION (Direction-sensitive)")
    print(f"{'='*60}")

    predicted_pairs_orig = create_relation_pairs_original(pred_df)
    ground_truth_pairs_orig = create_relation_pairs_original(gt_equal_df)

    print(f"Predicted unique pairs: {len(predicted_pairs_orig)}")
    print(f"Ground truth unique pairs: {len(ground_truth_pairs_orig)}")

    true_positives_orig = predicted_pairs_orig.intersection(ground_truth_pairs_orig)
    false_positives_orig = predicted_pairs_orig - ground_truth_pairs_orig
    false_negatives_orig = ground_truth_pairs_orig - predicted_pairs_orig

    tp_orig = len(true_positives_orig)
    fp_orig = len(false_positives_orig)
    fn_orig = len(false_negatives_orig)

    print("\nConfusion Matrix (Original):")
    print(f"True Positives (TP): {tp_orig}")
    print(f"False Positives (FP): {fp_orig}")
    print(f"False Negatives (FN): {fn_orig}")

    precision_orig = tp_orig / (tp_orig + fp_orig) if (tp_orig + fp_orig) > 0 else 0.0
    recall_orig = tp_orig / (tp_orig + fn_orig) if (tp_orig + fn_orig) > 0 else 0.0
    f1_score_orig = (
        2 * (precision_orig * recall_orig) / (precision_orig + recall_orig)
        if (precision_orig + recall_orig) > 0
        else 0.0
    )

    print("\nMetrics (Original):")
    print(f"Precision: {precision_orig:.4f} ({tp_orig}/{tp_orig + fp_orig})")
    print(f"Recall: {recall_orig:.4f} ({tp_orig}/{tp_orig + fn_orig})")
    print(f"F1-Score: {f1_score_orig:.4f}")

    print(f"\n{'='*60}")
    print("SYMMETRIC EVALUATION (Direction-insensitive)")
    print(f"{'='*60}")

    predicted_pairs_norm = create_relation_pairs_normalized(pred_df)
    ground_truth_pairs_norm = create_relation_pairs_normalized(gt_equal_df)

    print(f"Predicted unique normalized pairs: {len(predicted_pairs_norm)}")
    print(f"Ground truth unique normalized pairs: {len(ground_truth_pairs_norm)}")

    true_positives_norm = predicted_pairs_norm.intersection(ground_truth_pairs_norm)
    false_positives_norm = predicted_pairs_norm - ground_truth_pairs_norm
    false_negatives_norm = ground_truth_pairs_norm - predicted_pairs_norm

    tp_norm = len(true_positives_norm)
    fp_norm = len(false_positives_norm)
    fn_norm = len(false_negatives_norm)

    print("\nConfusion Matrix (Symmetric):")
    print(f"True Positives (TP): {tp_norm}")
    print(f"False Positives (FP): {fp_norm}")
    print(f"False Negatives (FN): {fn_norm}")

    precision_norm = tp_norm / (tp_norm + fp_norm) if (tp_norm + fp_norm) > 0 else 0.0
    recall_norm = tp_norm / (tp_norm + fn_norm) if (tp_norm + fn_norm) > 0 else 0.0
    f1_score_norm = (
        2 * (precision_norm * recall_norm) / (precision_norm + recall_norm)
        if (precision_norm + recall_norm) > 0
        else 0.0
    )

    print("\nMetrics (Symmetric):")
    print(f"Precision: {precision_norm:.4f} ({tp_norm}/{tp_norm + fp_norm})")
    print(f"Recall: {recall_norm:.4f} ({tp_norm}/{tp_norm + fn_norm})")
    print(f"F1-Score: {f1_score_norm:.4f}")

    precision_improvement = precision_norm - precision_orig
    recall_improvement = recall_norm - recall_orig
    f1_improvement = f1_score_norm - f1_score_orig

    print(f"\n{'='*60}")
    print("IMPROVEMENT (Symmetric vs Original)")
    print(f"{'='*60}")
    print(f"Precision improvement: {precision_improvement:+.4f}")
    print(f"Recall improvement: {recall_improvement:+.4f}")
    print(f"F1-Score improvement: {f1_improvement:+.4f}")

    llm_metrics = None
    if enable_llm_evaluation and len(false_positives_norm) > 0:
        print(f"\n{'='*60}")
        print("LLM-BASED RE-EVALUATION")
        print(f"{'='*60}")

        llm_eval_results = llm_evaluate_false_positives(
            pred_df, gt_equal_df, false_positives_norm, output_folder, llm_confidence_threshold, llm_max_evaluations
        )

        if llm_eval_results:
            llm_tp_set = set(llm_eval_results["llm_true_positives"])

            adjusted_tp = tp_norm + len(llm_tp_set)
            adjusted_fp = fp_norm - len(llm_tp_set)
            adjusted_fn = fn_norm

            adjusted_precision = adjusted_tp / (adjusted_tp + adjusted_fp) if (adjusted_tp + adjusted_fp) > 0 else 0.0
            adjusted_recall = adjusted_tp / (adjusted_tp + adjusted_fn) if (adjusted_tp + adjusted_fn) > 0 else 0.0
            adjusted_f1 = (
                2 * (adjusted_precision * adjusted_recall) / (adjusted_precision + adjusted_recall)
                if (adjusted_precision + adjusted_recall) > 0
                else 0.0
            )

            print(f"\n{'='*60}")
            print("ADJUSTED METRICS (After LLM Re-evaluation)")
            print(f"{'='*60}")
            print(f"Adjusted Precision: {adjusted_precision:.4f} ({adjusted_tp}/{adjusted_tp + adjusted_fp})")
            print(f"Adjusted Recall: {adjusted_recall:.4f} ({adjusted_tp}/{adjusted_tp + adjusted_fn})")
            print(f"Adjusted F1-Score: {adjusted_f1:.4f}")

            print("\nImprovement by LLM evaluation:")
            print(f"Precision improvement: {adjusted_precision - precision_norm:+.4f}")
            print(f"Recall improvement: {adjusted_recall - recall_norm:+.4f}")
            print(f"F1-Score improvement: {adjusted_f1 - f1_score_norm:+.4f}")

            llm_metrics = {
                "precision": adjusted_precision,
                "recall": adjusted_recall,
                "f1_score": adjusted_f1,
                "true_positives": adjusted_tp,
                "false_positives": adjusted_fp,
                "false_negatives": adjusted_fn,
                "llm_identified_tp": len(llm_tp_set),
                "total_evaluated": llm_eval_results["total_evaluated"],
            }

    print(f"\n{'='*60}")
    print("DATASET-WISE ANALYSIS (Symmetric)")
    print(f"{'='*60}")

    pred_dataset_stats = {}
    for pair in predicted_pairs_norm:
        ds_pair = tuple(sorted([pair[0], pair[2]]))
        if ds_pair not in pred_dataset_stats:
            pred_dataset_stats[ds_pair] = 0
        pred_dataset_stats[ds_pair] += 1

    gt_dataset_stats = {}
    for pair in ground_truth_pairs_norm:
        ds_pair = tuple(sorted([pair[0], pair[2]]))
        if ds_pair not in gt_dataset_stats:
            gt_dataset_stats[ds_pair] = 0
        gt_dataset_stats[ds_pair] += 1

    tp_dataset_stats = {}
    for pair in true_positives_norm:
        ds_pair = tuple(sorted([pair[0], pair[2]]))
        if ds_pair not in tp_dataset_stats:
            tp_dataset_stats[ds_pair] = 0
        tp_dataset_stats[ds_pair] += 1

    all_datasets = set(pred_dataset_stats.keys()) | set(gt_dataset_stats.keys())

    print("Dataset Pair | Predicted | Ground Truth | True Positives | Precision| Recall")
    print("-" * 80)
    for ds_pair in sorted(all_datasets):
        pred_count = pred_dataset_stats.get(ds_pair, 0)
        gt_count = gt_dataset_stats.get(ds_pair, 0)
        tp_count = tp_dataset_stats.get(ds_pair, 0)

        ds_precision = tp_count / pred_count if pred_count > 0 else 0.0
        ds_recall = tp_count / gt_count if gt_count > 0 else 0.0

        if ds_pair[0] == ds_pair[1]:
            ds_name = f"Within {ds_pair[0]}"
        else:
            ds_name = f"{ds_pair[0]} <-> {ds_pair[1]}"

        print(
            f"{ds_name:<20} | {pred_count:>9} | {gt_count:>12} | {tp_count:>14} | {ds_precision:>9.3f} | {ds_recall:>6.3f}"
        )

    return {
        "original": {
            "precision": precision_orig,
            "recall": recall_orig,
            "f1_score": f1_score_orig,
            "true_positives": tp_orig,
            "false_positives": fp_orig,
            "false_negatives": fn_orig,
        },
        "symmetric": {
            "precision": precision_norm,
            "recall": recall_norm,
            "f1_score": f1_score_norm,
            "true_positives": tp_norm,
            "false_positives": fp_norm,
            "false_negatives": fn_norm,
        },
        "improvement": {
            "precision": precision_improvement,
            "recall": recall_improvement,
            "f1_score": f1_improvement,
        },
        "llm_adjusted": llm_metrics,
    }


def main():
    predicted_file = AUTO_METAVD_DIR / f"auto_metavd_{VECTORIZATION}_{THRESHOLD}.csv"
    ground_truth_file = METAVD_DIR / "metavd_v1.csv"
    output_folder = AUTO_METAVD_DIR

    enable_llm_evaluation = True
    llm_confidence_threshold = 0.7
    llm_max_evaluations = None

    if not predicted_file.exists():
        print(f"Error: Predicted file not found: {predicted_file}")
        print("Please run extract_equal_relations.py first to generate the relations.")
        return

    if ground_truth_file.exists():
        print(f"\n{'='*60}")
        print("EVALUATION AGAINST GROUND TRUTH")
        print(f"{'='*60}")

        metrics = calculate_metrics(
            predicted_file,
            ground_truth_file,
            output_folder=output_folder,
            enable_llm_evaluation=enable_llm_evaluation,
            llm_confidence_threshold=llm_confidence_threshold,
            llm_max_evaluations=llm_max_evaluations,
        )

        if metrics:
            results_file = Path(output_folder) / f"evaluation_results_detailed_{VECTORIZATION}_{THRESHOLD}.txt"
            with results_file.open("w", encoding="utf-8") as f:
                f.write("ConceptNet-based Action Relation Extraction Evaluation\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Predicted file: {predicted_file}\n")
                f.write(f"Ground truth file: {ground_truth_file}\n\n")

                f.write("ORIGINAL EVALUATION (Direction-sensitive)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision: {metrics['original']['precision']:.4f}\n")
                f.write(f"Recall: {metrics['original']['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['original']['f1_score']:.4f}\n")
                f.write(f"True Positives: {metrics['original']['true_positives']}\n")
                f.write(f"False Positives: {metrics['original']['false_positives']}\n")
                f.write(f"False Negatives: {metrics['original']['false_negatives']}\n\n")

                f.write("SYMMETRIC EVALUATION (Direction-insensitive)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision: {metrics['symmetric']['precision']:.4f}\n")
                f.write(f"Recall: {metrics['symmetric']['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['symmetric']['f1_score']:.4f}\n")
                f.write(f"True Positives: {metrics['symmetric']['true_positives']}\n")
                f.write(f"False Positives: {metrics['symmetric']['false_positives']}\n")
                f.write(f"False Negatives: {metrics['symmetric']['false_negatives']}\n\n")

                f.write("IMPROVEMENT\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision improvement: {metrics['improvement']['precision']:+.4f}\n")
                f.write(f"Recall improvement: {metrics['improvement']['recall']:+.4f}\n")
                f.write(f"F1-Score improvement: {metrics['improvement']['f1_score']:+.4f}\n\n")

                if metrics.get("llm_adjusted"):
                    f.write("LLM ADJUSTED EVALUATION\n")
                    f.write("-" * 40 + "\n")
                    llm_metrics = metrics["llm_adjusted"]
                    f.write(f"Total evaluated by LLM: {llm_metrics['total_evaluated']}\n")
                    f.write(f"LLM identified True Positives: {llm_metrics['llm_identified_tp']}\n\n")
                    f.write(f"Adjusted Precision: {llm_metrics['precision']:.4f}\n")
                    f.write(f"Adjusted Recall: {llm_metrics['recall']:.4f}\n")
                    f.write(f"Adjusted F1-Score: {llm_metrics['f1_score']:.4f}\n")
                    f.write(f"Adjusted True Positives: {llm_metrics['true_positives']}\n")
                    f.write(f"Adjusted False Positives: {llm_metrics['false_positives']}\n")
                    f.write(f"Adjusted False Negatives: {llm_metrics['false_negatives']}\n\n")

                    precision_improvement_llm = llm_metrics["precision"] - metrics["symmetric"]["precision"]
                    recall_improvement_llm = llm_metrics["recall"] - metrics["symmetric"]["recall"]
                    f1_improvement_llm = llm_metrics["f1_score"] - metrics["symmetric"]["f1_score"]

                    f.write("IMPROVEMENT BY LLM EVALUATION\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Precision improvement: {precision_improvement_llm:+.4f}\n")
                    f.write(f"Recall improvement: {recall_improvement_llm:+.4f}\n")
                    f.write(f"F1-Score improvement: {f1_improvement_llm:+.4f}\n")

            print(f"\nDetailed evaluation results saved to: {results_file}")
    else:
        print(f"Warning: Ground truth file not found: {ground_truth_file}")


if __name__ == "__main__":
    main()
