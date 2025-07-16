 

import sys
import os
import re
import pandas as pd

def extract_bleu_score(file_path):
    """Extract BLEU score from fairseq output file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for BLEU score in the output
        bleu_match = re.search(r'BLEU = ([\d.]+)', content)
        if bleu_match:
            return float(bleu_match.group(1))
        else:
            print(f"No BLEU score found in {file_path}")
            return 0.0
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0.0
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0.0

def main():
    if len(sys.argv) < 4:
        print("Usage: python make_table.py METHOD ID ROOT_PATH [languages...]")
        sys.exit(1)
    
    method = sys.argv[1]
    exp_id = sys.argv[2]
    root_path = sys.argv[3]
    
    # Default languages if not provided
    if len(sys.argv) > 4:
        languages = sys.argv[4:]
    else:
        languages = ["en", "de", "es", "it"]
    
    # Define language pairs
    language_pairs = [
        ("en", "de"), ("de", "en"),
        ("en", "es"), ("es", "en"), 
        ("en", "it"), ("it", "en")
    ]
    
    results_dir = os.path.join(root_path, "pdec_work", "results", method, exp_id)
    
    # Collect results
    results = {}
    
    for src, tgt in language_pairs:
        file_path = os.path.join(results_dir, f"europarl_{src}_{tgt}.txt")
        bleu_score = extract_bleu_score(file_path)
        results[f"{src}-{tgt}"] = bleu_score
        print(f"{src} -> {tgt}: BLEU = {bleu_score:.2f}")
    
    # Create summary table
    df_data = []
    
    # Supervised translation results (X -> en and en -> X)
    supervised_pairs = []
    for lang in ["de", "es", "it"]:
        supervised_pairs.extend([(lang, "en"), ("en", lang)])
    
    for src, tgt in supervised_pairs:
        pair_key = f"{src}-{tgt}"
        if pair_key in results:
            df_data.append({
                "Language Pair": f"{src} → {tgt}",
                "BLEU Score": results[pair_key],
                "Type": "Supervised"
            })
    
    # Calculate averages
    supervised_scores = [results[f"{src}-{tgt}"] for src, tgt in supervised_pairs if f"{src}-{tgt}" in results]
    avg_supervised = sum(supervised_scores) / len(supervised_scores) if supervised_scores else 0.0
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Europarl Results Summary - {method} (ID: {exp_id})")
    print("="*50)
    print(df.to_string(index=False))
    print(f"\nAverage BLEU (Supervised): {avg_supervised:.2f}")
    
    # Save to Excel if pandas and openpyxl are available
    try:
        excel_dir = os.path.join(root_path, "pdec_work", "excel")
        os.makedirs(excel_dir, exist_ok=True)
        excel_path = os.path.join(excel_dir, f"europarl_{method}_{exp_id}_results.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\nResults saved to: {excel_path}")
    except ImportError:
        print("\nNote: Install openpyxl to save Excel files")
        # Save as CSV instead
        csv_path = os.path.join(excel_dir, f"europarl_{method}_{exp_id}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()