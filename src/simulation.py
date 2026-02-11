
import json
import asyncio
import random
import yaml
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm
from src.utils import get_openai_key, get_anthropic_key

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")


def _get_project_root():
    """return the project root"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root


def _load_prompts():
    """load prompts from yaml file"""
    project_root = _get_project_root()
    prompts_file = project_root / "resources" / "prompt" / "simulator_prompts.yaml"
    
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    return prompts


PROMPTS = _load_prompts()

MAX_CONCURRENT_REQUESTS = 10
client = AsyncOpenAI(api_key=get_openai_key())


async_anthropic_client = None

if ANTHROPIC_AVAILABLE:
    try:
        anthropic_key = get_anthropic_key()
        if anthropic_key and anthropic_key.strip():
            if not anthropic_key.startswith("sk-ant-"):
                print(f"Warning: Anthropic API key format may be incorrect (expected 'sk-ant-...'). Got: {anthropic_key[:10]}...")
            async_anthropic_client = AsyncAnthropic(api_key=anthropic_key.strip())
            print("Anthropic client initialized successfully")
        else:
            print("  Warning: Anthropic API key is empty or None. Anthropic models will not work.")
            print("  Please check conf.d/conf.yaml and ensure 'anthropic.key' is set correctly.")
    except KeyError as e:
        print(f"  Error: {e}")
        print("  Please add 'anthropic.key' to conf.d/conf.yaml")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
    except Exception as e:
        print(f"  Warning: Failed to initialize Anthropic client: {e}")
        print("  Anthropic models will not work. Please check conf.d/conf.yaml")

def is_anthropic_model(model_name):
    return model_name.startswith("claude-")

def is_openai_model(model_name):
    return model_name.startswith("gpt-")

async def call_llm(model_name, messages, system_prompt=None, temperature=0.0):
    if is_anthropic_model(model_name):
        if not ANTHROPIC_AVAILABLE:
            raise ValueError(f"Anthropic package not installed. Install with: pip install anthropic")
        if async_anthropic_client is None:
            raise ValueError(f"Anthropic client not initialized. Please check your API key in conf.d/conf.yaml")

        anthropic_messages = []
        extracted_system_prompt = None
        
        if system_prompt is None:
            for msg in messages:
                if msg["role"] == "system":
                    extracted_system_prompt = msg["content"]
                elif msg["role"] in ["user", "assistant"]:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        else:
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        final_system_prompt = system_prompt if system_prompt else extracted_system_prompt
        
        response = await async_anthropic_client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=final_system_prompt if final_system_prompt else "",
            messages=anthropic_messages,
            temperature=temperature
        )
        return response.content[0].text
    else:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.0 if model_name=="gpt-5" else temperature
        )
        return response.choices[0].message.content

def format_options(options_dict):
    text = ""
    for key, value in options_dict.items():
        text += f"({key}) {value}\n"
    return text

async def run_single_simulation(target_model, case, mode, semaphore, simulator_model):
    async with semaphore:
        try:
            if mode == "error":
                if not case.get('distractors'): return None
                target_belief = random.choice(case['distractors'])
            elif mode == "correct":
                target_belief = case['options'][case['correct_option']]

            simulator_sys_prompt_turn_1 = PROMPTS["simulator_base_system_prompt_turn_1"].format(
                target_belief=target_belief
            )

            simulator_sys_prompt_turn_2 = PROMPTS["simulator_base_system_prompt_turn_2"].format(
                target_belief=target_belief
            )

            target_sys_prompt = PROMPTS["target_ai_system_prompt"]


            scenario_text = case['scenario']
            image_caption = case['caption']
            history = []

            # --- Dialogue Loop (2 Turns) ---
            
            # [Turn 1] Simulator
            sim_msg_1 = await call_llm(
                model_name=simulator_model,
                messages=[
                    {"role": "system", "content": simulator_sys_prompt_turn_1},
                    {"role": "user", "content": PROMPTS["user_messages"]["simulator_turn1_user"].format(
                        scenario_text=scenario_text, 
                        target_belief=target_belief,
                        image_caption=image_caption
                    )}
                ],
                temperature=0.0
            )
            history.append({"role": "Doctor", "content": sim_msg_1})

            # [Turn 1] Target AI
            ai_msg_1 = await call_llm(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {"role": "user", "content": PROMPTS["user_messages"]["ai_turn1_user"].format(
                        scenario_text=scenario_text, 
                        image_caption=image_caption,
                        sim_msg_1=sim_msg_1
                    )}
                ],
                temperature=0.0
            )
            history.append({"role": "AI", "content": ai_msg_1})

            # [Turn 2] Simulator (Rebuttal)
            sim_msg_2 = await call_llm(
                model_name=simulator_model,
                messages=[
                    {"role": "system", "content": simulator_sys_prompt_turn_2},
                    {"role": "user", "content": PROMPTS["user_messages"]["simulator_turn2_user"].format(
                        scenario_text=scenario_text,
                        image_caption=image_caption,
                        options_text=format_options(case['options']),
                        sim_msg_1=sim_msg_1,
                        ai_msg_1=ai_msg_1,
                        target_belief=target_belief
                    )}
                ],
                temperature=0.0
            )
            
            history.append({"role": "Doctor", "content": sim_msg_2})

            # [Turn 2] Target AI (Final Response)
            ai_msg_2 = await call_llm(
                model_name=target_model,
                messages=[
                    {"role": "system", "content": target_sys_prompt},
                    {
                        "role": "user",
                        "content": PROMPTS["user_messages"]["ai_turn2_user"].format(
                            scenario_text=scenario_text,
                            image_caption=image_caption,
                            sim_msg_1=sim_msg_1,
                            ai_msg_1=ai_msg_1,
                            sim_msg_2=sim_msg_2,
                        ),
                    },
                ],
                temperature=0.0,
            )
            history.append({"role": "AI", "content": ai_msg_2})

            return {
                "case_id": case.get("case_id"),
                "mode": mode,
                "dialogue": history,
                "ground_truth": case['correct_option'],
                "target_belief": target_belief
            }

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg.lower():
                print(f"   Error in {target_model} (Case {case.get('case_id')}): Model not found (404)")
                print(f"   This model may be deprecated or the name is incorrect.")
                print(f"   Please check Anthropic documentation for available model names.")
            else:
                print(f"Error in {target_model} (Case {case.get('case_id')}): {e}")
            return None

async def simulate_two_modes(model_name, dataset, semaphore, simulator_model):
    print(f"Starting evaluation for model: {model_name}")
    
    tasks = []
    for case in dataset:
        tasks.append(run_single_simulation(model_name, case, "error", semaphore, simulator_model))
        tasks.append(run_single_simulation(model_name, case, "correct", semaphore, simulator_model))
    
    logs = []
    with tqdm(total=len(tasks), desc=f"[{model_name}]", leave=False) as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res: logs.append(res)
            pbar.update(1)
    
    print(f"\n  {model_name} completed: {len(logs)} logs generated")
    
    return {
        "model": model_name,
        "logs": logs
    }


async def run_simulation(
    input_file: str,
    output_file: str,
    target_models: list,
    simulator_model: str = "gpt-4o",
    max_concurrent_requests: int = 10
):
    project_root = _get_project_root()
    
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_file
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = project_root / output_file
    
    print(f"  Loading data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                combined_results = json.load(f)
            print(f"  Loaded existing results: {len(combined_results)} models already evaluated")
        except Exception as e:
            print(f"   Warning: Could not load existing results file: {e}")
            combined_results = {}
    
    remaining_models = [model for model in target_models if model not in combined_results]
    if not remaining_models:
        print("  All models already evaluated!")
        if combined_results:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(combined_results, f, indent=4, ensure_ascii=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
        return
    
    print(f"\n Starting parallel evaluation for {len(remaining_models)} remaining models")
    print(f"   Total cases: {len(dataset)}")
    print(f"   Total simulations: {len(dataset) * 2 * len(remaining_models)} (2 modes × {len(remaining_models)} models)")
    print(f"   Max concurrent requests: {max_concurrent_requests}")
    print(f"   Results will be saved to: {output_path}")
    print("")
    
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    tasks = {
        asyncio.create_task(simulate_two_modes(model, dataset, semaphore, simulator_model)): model
        for model in remaining_models
    }
    
    completed_count = 0
    total_models = len(remaining_models)
    
    for coro in asyncio.as_completed(tasks.keys()):
        result = await coro
        model_name = result["model"]
        
        model_result = {k: v for k, v in result.items() if k != "model"}
        combined_results[model_name] = model_result
        
        completed_count += 1
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=4, ensure_ascii=False)
        
        print(f"    Progress saved ({completed_count}/{total_models} models completed)")
    
    print(f"\n All evaluations complete! {len(combined_results)} model results saved to '{output_path}'")