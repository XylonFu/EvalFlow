import os
import json
import csv
from collections import defaultdict

base_path = "/gpfs/work/int/xinlongfu24/xinlong_fu/scripts/results/evalscope/Qwen3-4B-RL/reports"

data = defaultdict(dict)
all_models = []
all_datasets = set()

for model_dir in os.listdir(base_path):
    model_path = os.path.join(base_path, model_dir)
    if not os.path.isdir(model_path):
        continue
    
    for version_dir in os.listdir(model_path):
        version_path = os.path.join(model_path, version_dir)
        if not os.path.isdir(version_path):
            continue
            
        full_model_name = f"{model_dir}/{version_dir}"
        all_models.append(full_model_name)
        
        json_files = [f for f in os.listdir(version_path) if f.endswith(".json")]
        
        for json_file in json_files:
            json_path = os.path.join(version_path, json_file)
            try:
                with open(json_path, 'r') as f:
                    content = json.load(f)
                    dataset = content.get('dataset_pretty_name', content.get('dataset_pretty_name', 'N/A'))
                    score = content.get('score', 'N/A')
                    
                    if dataset != 'N/A' and score != 'N/A':
                        data[dataset][full_model_name] = score
                        all_datasets.add(dataset)
            except Exception:
                continue

def sort_model_name(name):
    parts = name.split('/')
    model_part = parts[0]
    version_part = parts[1] if len(parts) > 1 else ''
    try:
        version_num = int(version_part.split('_')[-1]) if version_part and version_part.split('_')[-1].isdigit() else 0
    except:
        version_num = 0
    return (model_part, version_num)

sorted_models = sorted(all_models, key=sort_model_name)

headers = ["Model/Dataset"] + sorted(all_datasets) + ["Avg."]
rows = []

for model in sorted_models:
    row = [model]
    valid_scores = []
    
    for dataset in headers[1:-1]:
        score = data[dataset].get(model, 'N/A')
        row.append(score)
        if score != 'N/A':
            valid_scores.append(float(score))
    
    avg_score = sum(valid_scores)/len(valid_scores) if valid_scores else 'N/A'
    if avg_score != 'N/A':
        avg_score = round(avg_score, 4)
    row.append(avg_score)
    rows.append(row)

output_file = "evalscope_scores.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"Saved to {output_file}")
print(f"Models: {', '.join(sorted_models)}")
print(f"Datasets: {', '.join(sorted(all_datasets))}")
