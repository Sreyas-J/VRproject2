import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import gdown
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel


def download(output_dir, google_drive_url):
    """Download model weights from Google Drive if not already present."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Downloading model weights from Google Drive to {output_dir}...")
        gdown.download_folder(url=google_drive_url, output=output_dir, quiet=False)
    else:
        print(f"Model weights already exist at {output_dir}")


def load_finetuned_model(model_dir):
    """Load the fine-tuned BLIP model with LoRA adapter."""
    MODEL_NAME = "Salesforce/blip-vqa-capfilt-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load processor (feature extractor + tokenizer)
    #processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
	
    # 2. Load the base BLIP model
    base_model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)

    # 3. Load the LoRA adapter and apply it to the base model
    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        is_trainable=False,
        local_files_only=True
    )
    model.eval()

    return processor, model, device


def run_inference(image_dir: str, csv_path: str, model_dir: str):
    """Load model, run inference on all rows in CSV, and return list of generated answers."""
    # Download model weights if needed (uncomment to enable)
    # download(model_dir, "https://drive.google.com/drive/folders/1nEkAdYeWluPJ4k5_6cyWPGAFf0fnqbXs?usp=sharing")

    # Load processor, model, and device
    processor, model, device = load_finetuned_model(model_dir)

    # Load metadata CSV
    df = pd.read_csv(csv_path)
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(image_dir, row['image_name'])
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=question, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                answer = processor.decode(generated_ids[0], skip_special_tokens=True)
            # Post-process answer to be one word and lowercase
            answer = str(answer).split()[0].lower()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            answer = "error"
        generated_answers.append(answer)

    # Add generated answers to DataFrame and save to results.csv
    df["generated_answer"] = generated_answers
    output_csv = "results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to {output_csv}")
    return generated_answers


def main():
    parser = argparse.ArgumentParser(description="Run BLIP-VQA LoRA inference on a set of images and questions.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--model_dir', type=str, default='./gdrive/blip-vqa-adapters', help='Path to downloaded model weights')
    args = parser.parse_args()

    # Run inference and optionally print results list
    answers = run_inference(args.image_dir, args.csv_path, args.model_dir)
    print(answers)


if __name__ == '__main__':
    main()