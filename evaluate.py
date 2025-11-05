import os, json, time
from colorama import Fore, Style, init
from genai_client import (
    initialize_genai_client, generate_video_from_image, evaluate_video
)

NUM_VIDEOS_PER_ITEM = 6
OUTPUT_JSON = "mme-cof_eval_results.json"

def process_json_data(data_file: str = "data.json"):
    """
    Process data in a JSON file: generate 6 videos per item and evaluate each
    """
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Unified video generation & evaluation pipeline - {NUM_VIDEOS_PER_ITEM} per item (16:9 padded){Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")

    # Check data file
    if not os.path.exists(data_file):
        print(f"Data file does not exist: {data_file}")
        return

    # Initialize GenAI client
    genai_client = initialize_genai_client()
    if not genai_client:
        print("Unable to initialize video generation client; skipping generation step")

    # Read data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📋 Found {len(data)} items; each will generate {NUM_VIDEOS_PER_ITEM} videos (16:9 format)")

    results = []

    for i, item in enumerate(data):
        if "image" not in item or "reasoning_prompt" not in item:
            print(f"Item {i+1} missing required fields, skipping")
            continue

        print(f"\n{Fore.BLUE}[{i+1}/{len(data)}] Processing item: {item['image']}{Style.RESET_ALL}")
        print(f"Prompt: {item["reasoning_prompt"][:100]}...")

        image_path = item["image"]
        prompt = item["reasoning_prompt"]

        # Create output directory
        base_name = os.path.splitext(image_path)[0]
        output_dir = f"{base_name}_videos"
        os.makedirs(output_dir, exist_ok=True)

        item_result = item.copy()
        item_result["output_directory"] = output_dir
        item_result["generated_videos"] = []
        item_result["evaluation_results"] = {}
        item_result["generation_attempts"] = NUM_VIDEOS_PER_ITEM
        item_result["padding_applied"] = "16:9"  # Record padding info

        # Generate multiple videos
        successful_generations = 0
        generation_errors = []

        if genai_client and os.path.exists(image_path):
            for video_num in range(1, NUM_VIDEOS_PER_ITEM + 1):
                print(f"\n{Fore.CYAN}=== Generating video {video_num}/{NUM_VIDEOS_PER_ITEM} (16:9 Padded) ==={Style.RESET_ALL}")

                video_path = os.path.join(output_dir, f"generated_video_{video_num}.mp4")

                # Multiple attempts per video
                max_retries = 5
                success = False

                if not os.path.exists(video_path):
                    for attempt in range(1, max_retries + 1):
                        print(f"🎬 Video #{video_num} generation attempt {attempt}/{max_retries}")

                        success = generate_video_from_image(genai_client, image_path, prompt, video_path, video_num)

                        if success:
                            print(f"Video #{video_num} succeeded on attempt {attempt}!")
                            break
                        else:
                            print(f"Video #{video_num} failed on attempt {attempt}")
                            if attempt < max_retries:
                                retry_wait = 30
                                print(f"⏳ Waiting {retry_wait} seconds before retry...")
                                time.sleep(retry_wait)
                else:
                    success = True

                # Handle generation result
                if success:
                    item_result["generated_videos"].append(video_path)
                    successful_generations += 1
                    print(f"Video #{video_num} generated: {video_path}")

                    # Small delay before evaluation
                    print(f"⏳ Waiting 2 seconds before evaluating video #{video_num}...")
                    time.sleep(2)

                    # Evaluate the generated video
                    evaluation_result = evaluate_video(video_path, prompt, video_num)

                    # Add None check to prevent errors
                    if evaluation_result is None:
                        print(f"!  Warning: Evaluation returned None for video #{video_num}")
                        evaluation_result = f"error: Evaluation returned None for video #{video_num}"

                    item_result["evaluation_results"][video_path] = {
                        "video_id": video_num,
                        "evaluation": evaluation_result,
                        "status": "success" if evaluation_result and not evaluation_result.startswith("error:") else "failed"
                    }

                    print(f"Evaluation for video #{video_num} completed")

                    # Rest between videos
                    if video_num < NUM_VIDEOS_PER_ITEM:
                        wait_time = 10
                        print(f"⏳ Waiting {wait_time} seconds before generating the next video...")
                        time.sleep(wait_time)

                else:
                    # All retries failed
                    error_msg = f"Video #{video_num} generation failed after {max_retries} attempts"
                    print(f"{error_msg}")
                    generation_errors.append(error_msg)

                    # Even after failure, wait a bit before moving on
                    if video_num < NUM_VIDEOS_PER_ITEM:
                        wait_time = 30
                        print(f"⏳ Waiting {wait_time} seconds before continuing to the next video...")
                        time.sleep(wait_time)

        else:
            if not genai_client:
                error_msg = "GenAI client not available"
                print(f"{error_msg}")
                generation_errors.append(error_msg)
            elif not os.path.exists(image_path):
                error_msg = f"Image file not found: {image_path}"
                print(f"{error_msg}")
                generation_errors.append(error_msg)

        # Record generation errors
        if generation_errors:
            item_result["generation_errors"] = generation_errors

        # If there are existing videos, evaluate them too
        existing_videos = []
        if os.path.exists(output_dir):
            existing_videos = [f for f in os.listdir(output_dir)
                               if f.endswith('.mp4') and not f.startswith("generated_video_")]

        for video_file in existing_videos:
            video_path = os.path.join(output_dir, video_file)
            print(f"📹 Found existing video, evaluating: {video_file}")

            evaluation_result = evaluate_video(video_path, prompt, 0)
            item_result["evaluation_results"][video_path] = {
                "video_id": 0,
                "evaluation": evaluation_result,
                "status": "success" if not evaluation_result.startswith("error:") else "failed"
            }

        # Add statistics
        successful_evaluations = len([r for r in item_result["evaluation_results"].values() if r["status"] == "success"])
        total_evaluations = len(item_result["evaluation_results"])

        item_result["summary"] = {
            "videos_requested": NUM_VIDEOS_PER_ITEM,
            "videos_generated": successful_generations,
            "generation_success_rate": f"{successful_generations/NUM_VIDEOS_PER_ITEM*100:.1f}%",
            "videos_evaluated": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "evaluation_success_rate": f"{successful_evaluations/max(total_evaluations, 1)*100:.1f}%"
        }

        results.append(item_result)

        # Save intermediate results
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"\nItem {i+1} done:")
        print(f"   - Generated successfully: {successful_generations}/{NUM_VIDEOS_PER_ITEM} videos")
        print(f"   - Evaluated successfully: {successful_evaluations}/{total_evaluations} videos")
        print(f"   - Intermediate results saved")

        # Rest between items
        if i < len(data) - 1:
            wait_time = 10
            print(f"⏳ Waiting {wait_time} seconds before processing the next item...")
            time.sleep(wait_time)

    # Build final summary
    total_items = len(results)
    total_videos_requested = sum(r.get("generation_attempts", NUM_VIDEOS_PER_ITEM) for r in results)
    total_videos_generated = sum(len(r["generated_videos"]) for r in results)
    total_evaluations = sum(len(r["evaluation_results"]) for r in results)
    successful_evaluations = sum(r["summary"]["successful_evaluations"] for r in results)

    final_summary = {
        "processing_summary": {
            "total_items": total_items,
            "videos_per_item": NUM_VIDEOS_PER_ITEM,
            "total_videos_requested": total_videos_requested,
            "total_videos_generated": total_videos_generated,
            "generation_success_rate": f"{total_videos_generated/total_videos_requested*100:.1f}%",
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "evaluation_success_rate": f"{successful_evaluations/max(total_evaluations, 1)*100:.1f}%",
            "padding_applied": "16:9 aspect ratio"
        },
        "results": results
    }

    # Save final results
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=4)

    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🎉 All processing complete!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total items: {total_items}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total video requests: {total_videos_requested} ({NUM_VIDEOS_PER_ITEM}x{total_items}){Style.RESET_ALL}")
    print(f"{Fore.GREEN}Videos generated successfully: {total_videos_generated} ({total_videos_generated/total_videos_requested*100:.1f}%){Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total evaluations: {total_evaluations}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Successful evaluations: {successful_evaluations} ({successful_evaluations/max(total_evaluations, 1)*100:.1f}%){Style.RESET_ALL}")
    print(f"{Fore.GREEN}Results saved to: {OUTPUT_JSON}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}All input images were automatically padded to 16:9{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")


if __name__ == "__main__":
    init()  # Initialize colorama
    process_json_data()
