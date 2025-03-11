import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_paths", nargs='+', required=True)
    parser.add_argument("--model_names", nargs='+', required=True)
    parser.add_argument("--tasks", nargs='+', required=True)
    parser.add_argument("--need_avg", action="store_true")
    args = parser.parse_args()
    
    scores = []
    for dict_path in args.dict_paths:
        cur_scores = {}
        with open(dict_path, "r") as f:
            data = json.load(f)
        for task in args.tasks:
            cur_scores[task] = data[task]["acc"]
        if args.need_avg:
            cur_scores["avg"] = sum(cur_scores.values()) / len(cur_scores)
        scores.append(cur_scores)

    # Generate LaTeX table
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{c|" + "c" * len(args.tasks) + "c}")
    print("\\toprule")
    
    # Header row
    header = ["Model"] + args.tasks
    if args.need_avg:
        header.append("avg")
    print(" & ".join(header) + " \\\\")
    print("\\midrule")
    
    # Content rows with bold maximum values and underlined second-best values
    for i, score_dict in enumerate(scores):
        row = [args.model_names[i]]
        for task in args.tasks + ["avg"] if args.need_avg else args.tasks:
            value = score_dict[task]
            task_scores = [s[task] for s in scores]
            
            # Sort scores in descending order
            sorted_scores = sorted(task_scores, reverse=True)
            
            if value == sorted_scores[0]:  # Best score (maximum)
                row.append(f"\\textbf{{{value*100:.1f}}}")
            elif len(sorted_scores) > 1 and value == sorted_scores[1]:  # Second-best score
                row.append(f"\\underline{{{value*100:.1f}}}")
            else:
                row.append(f"{value*100:.1f}")
        print(" & ".join(row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
