import pandas as pd


def calculate_pass_at_n(file_paths, index_field, judge_field, filter_field=None, filter_value=None):
    all_dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        all_dfs.append(df)
    
    if filter_field and filter_value is not None:
        sample_count = len(all_dfs[0][all_dfs[0][filter_field].astype(str).str.contains(filter_value)])
    else:
        sample_count = len(all_dfs[0])
    
    correct_indices = set()
    
    for df in all_dfs:
        if filter_field and filter_value is not None:
            filtered_df = df[df[filter_field].astype(str).str.contains(filter_value)]
        else:
            filtered_df = df
        
        for _, row in filtered_df.iterrows():
            index_val = row[index_field]
            judge_val = row[judge_field]
            
            if (judge_val == "TRUE") or (judge_val == 1):
                correct_indices.add(index_val)
    
    pass_count = len(correct_indices)
    accuracy = pass_count / sample_count if sample_count > 0 else 0.0
    
    return pass_count, accuracy, sample_count


if __name__ == "__main__":
    base_dir = "/gpfs/work/int/xinlongfu24/xinlong_fu/scripts/results/evalscope/InternVL3-2B-N/InternVL3-2B-C340S50-N"
    file_paths = [f"{base_dir}{i}/InternVL3-2B-C340S50-N{i}_MathVista_MINI_gpt-4o-mini.xlsx" for i in range(1, 5)]

    index_field = "index"
    judge_field = "hit"
    filter_field = None
    filter_value = None
    
    pass_count, accuracy, sample_count = calculate_pass_at_n(
        file_paths, index_field, judge_field, filter_field, filter_value
    )
    
    print(f"Pass@n总数: {pass_count}")
    print(f"过滤后的样本总数: {sample_count}")
    print(f"准确率: {accuracy:.4f} ({pass_count}/{sample_count})")
