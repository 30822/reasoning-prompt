#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import OrderedDict


def get_category_from_filename(filename):
    match = re.search(r'jama_raw_(.+)\.json', filename)
    if match:
        return match.group(1)
    return None


def add_case_id_to_case(case, case_id):
    case_id_str = f"{case_id:04d}"
    new_case = OrderedDict()
    new_case['case_id'] = case_id_str
    
    for key, value in case.items():
        new_case[key] = value
    
    return new_case


def process_json_file(file_path, base_id):
    print(f"Processing: {file_path.name} (starting from case_id {base_id:04d})")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    updated_cases = []
    for idx, case in enumerate(cases):
        case_id = base_id + idx
        updated_case = add_case_id_to_case(case, case_id)
        updated_cases.append(updated_case)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_cases, f, indent=4, ensure_ascii=False)
    
    print(f"  ✅ Added case_id {base_id:04d} to {base_id + len(cases) - 1:04d} ({len(cases)} cases)")
    
    return len(cases)


def main():
    script_dir = Path(__file__).parent
    
    json_files = sorted(script_dir.glob('jama_raw*.json'))
    
    if not json_files:
        print("❌ No jama_raw*.json files found!")
        return
    
    print(f"Found {len(json_files)} files:")
    for f in json_files:
        print(f"  - {f.name}")
    print()
    
    current_base_id = 0
    total_cases = 0
    
    for file_path in json_files:
        category = get_category_from_filename(file_path.name)
        num_cases = process_json_file(file_path, current_base_id)
        total_cases += num_cases
        current_base_id += 1000
    
    print()
    print(f"✅ All done! Processed {total_cases} cases across {len(json_files)} files.")


if __name__ == "__main__":
    main()

