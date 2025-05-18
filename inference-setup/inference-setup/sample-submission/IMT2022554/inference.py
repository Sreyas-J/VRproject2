import re
import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import gdown
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel


def finetune_answer(s):
    """Normalize answer for more accurate comparison."""
    # Define reversed number map (digits to words)
    number_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
    }
    
    # Convert digits to words
    for digit, word in number_map.items():
        s = re.sub(r'\b' + digit + r'\b', word, s.lower())
    
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation and extra whitespace
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processor
    MODEL_NAME = "Salesforce/blip-vqa-base"
    # Load the model
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
    model.to(device)
    model.eval()
    return processor, model, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV')
    args = parser.parse_args()

    # Google Drive URL for model weights
    GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1nEkAdYeWluPJ4k5_6cyWPGAFf0fnqbXs?usp=sharing"
    MODEL_DIR = "gdrive/"

    # Download model weights
    download(MODEL_DIR, GOOGLE_DRIVE_URL)

    # Load model and processor
    print("Loading the model BLIP with LoRA...")
    processor, model, device = load_finetuned_model(f"{MODEL_DIR}blip-vqa-adapters/")

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row['image_name'])
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
            print(f"Error processing image {image_path}: {str(e)}")
            answer = "error"
        answer = finetune_answer(answer)
        generated_answers.append(answer)

    # Add generated answers to DataFrame and save to results.csv
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)
    print("Inference complete. Results saved to results.csv")

if __name__ == "__main__":
    main()

