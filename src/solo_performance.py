import json
import re
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm
import time
from src.utils import get_openai_key, get_anthropic_key


try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")


def _get_project_root():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def _initialize_clients():
    async_client = AsyncOpenAI(api_key=get_openai_key())
    
    async_anthropic_client = None
    if ANTHROPIC_AVAILABLE:
        try:
            anthropic_key = get_anthropic_key()
            if anthropic_key and anthropic_key.strip():
                if not anthropic_key.startswith("sk-ant-"):
                    print(f"Warning: Anthropic API key format may be incorrect (expected 'sk-ant-...'). Got: {anthropic_key[:10]}...")
                async_anthropic_client = AsyncAnthropic(api_key=anthropic_key.strip())
                print("  Anthropic client initialized successfully")
            else:
                print("  Warning: Anthropic API key is empty or None. Anthropic models will not work.")
                print("   Please check conf.d/conf.yaml and ensure 'anthropic.key' is set correctly.")
        except KeyError as e:
            print(f"  Error: {e}")
            print("   Please add 'anthropic.key' to conf.d/conf.yaml")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Warning: Failed to initialize Anthropic client: {e}")
            print("   Anthropic models will not work. Please check conf.d/conf.yaml")
    
    return async_client, async_anthropic_client

def is_anthropic_model(model_name):
    return model_name.startswith("claude-")

def is_openai_model(model_name):
    return model_name.startswith("gpt-")


def format_options(options_dict):
    text = ""
    for key, value in options_dict.items():
        text += f"({key}) {value}\n"
    return text

def extract_answer_key(text):
    explicit_patterns = [
        r"Answer:\s*\(?([A-D])\)?",    
        r"The answer is\s*\(?([A-D])\)?", 
        r"Correct option is\s*\(?([A-D])\)?", 
        r"Option\s+([A-D])\s+is",            
        r"Select\s+([A-D])",             
        r"Choice\s+([A-D])",              
        r"\*\*([A-D])\*\*",              
        r"Final answer:\s*\(?([A-D])\)?",  
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    bracket_matches = re.findall(r'\(([A-D])\)', text)
    if bracket_matches:
        return bracket_matches[-1].upper()
    start_match = re.search(r'^([A-D])[\.\s]', text)
    if start_match:
        return start_match.group(1).upper()
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1).upper()
        
    return "Unknown"


async def solve_case_solo_async(
    model_name, 
    case, 
    case_index, 
    semaphore,
    async_client: AsyncOpenAI,
    async_anthropic_client
):
    async with semaphore:
        scenario = case['scenario']
        options_text = format_options(case['options'])
        
        system_prompt = """
        You are a medical expert taking a clinical board exam.
        Read the case and select the single best answer.
        Output ONLY the option letter (A, B, C, or D) corresponding to your choice.
        Do not provide explanations.
        """
        
        user_prompt = f"""
        [Clinical Case]
        {scenario}
        
        [Options]
        {options_text}
        
        Question: What is the most appropriate next step or diagnosis?
        """
        
        try:
            if is_anthropic_model(model_name):
                if not ANTHROPIC_AVAILABLE:
                    raise ValueError(f"Anthropic package not installed. Install with: pip install anthropic")
                if async_anthropic_client is None:
                    raise ValueError(f"Anthropic client not initialized. Please check your API key in conf.d/conf.yaml")
                
                response = await async_anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1.0 if "sonnet" in model_name.lower() else 0.0
                )
                content = response.content[0].text.strip()
            else:
                response = await async_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1.0 if model_name=="gpt-5" else 0.0
                )
                content = response.choices[0].message.content.strip()
            
            prediction = extract_answer_key(content)
            
            return {
                "raw_response": content,
                "prediction": prediction,
                "is_correct": prediction == case['correct_option'],
                "case_id": case.get("new_idx"),
                "case_index": case_index 
            }
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg.lower():
                print(f"  Error with {model_name} (case {case.get('new_idx', 'unknown')}): Model not found (404)")
                print(f"   This model may be deprecated or the name is incorrect.")
                print(f"   Please check Anthropic documentation for available model names.")
            else:
                print(f"Error with {model_name} (case {case.get('new_idx', 'unknown')}): {e}")
            return None


async def evaluate_model_async(
    model_name, 
    dataset, 
    semaphore,
    async_client: AsyncOpenAI,
    async_anthropic_client
):
    tasks = [
        solve_case_solo_async(model_name, case, idx, semaphore, async_client, async_anthropic_client)
        for idx, case in enumerate(dataset)
    ]

    results = []
    with tqdm(total=len(tasks), desc=f"Evaluating {model_name}") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            pbar.update(1)
    
    results.sort(key=lambda x: x.get('case_index', 0))
    return results

async def run_solo_performance(
    input_file: str,
    output_file: str,
    models_to_test: list,
    max_concurrent_requests: int = 10
):
    project_root = _get_project_root()
    
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_file
    
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = project_root / output_file
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"  Loading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    async_client, async_anthropic_client = _initialize_clients()
    
    all_results = {}
    
    print(f"Starting Solo Performance Evaluation on {len(dataset)} cases...")
    print(f"Using async parallel processing (max {max_concurrent_requests} concurrent requests)")
    print("")
    
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    for model in models_to_test:
        print(f"\nEvaluating Model: {model}")
        start_time = time.time()
        
        results = await evaluate_model_async(
            model, 
            dataset, 
            semaphore,
            async_client,
            async_anthropic_client
        )
    
        model_logs = []
        correct_count = 0
        
        for result in results:
            if result:
                idx = result['case_index']
                case = dataset[idx]  
                
                log_entry = {
                    "case_id": case.get("case_id"),
                    "ground_truth": case['correct_option'],
                    "prediction": result['prediction'],
                    "is_correct": result['is_correct'],
                    "raw_response": result['raw_response']
                }
                model_logs.append(log_entry)
                if result['is_correct']:
                    correct_count += 1
        
        elapsed_time = time.time() - start_time
        accuracy = correct_count / len(dataset) if dataset else 0
        print(f"Model: {model} | Accuracy: {accuracy:.2%} | Time: {elapsed_time:.1f}s")
        
        all_results[model] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_cases": len(dataset),
            "logs": model_logs
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n  All evaluations complete. Results saved to '{output_path}'")