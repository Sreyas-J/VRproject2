# Introduction
Multimodal Visual Question Answering (VQA) is a challenging research problem that lies at the intersection of computer vision and natural language processing. It involves building models capable of answering questions based on image inputs, demanding an understanding of both visual semantics and linguistic context. With increasing demand for AI systems that can interpret and reason over multimodal data, VQA has found applications in accessibility tools, educational technologies, and intelligent virtual assistants.
This project explores multimodal VQA using the Amazon Berkeley Objects (ABO) dataset, which consists of over 147,000 product listings and nearly 400,000 catalog images with multilingual metadata. The primary objective of this mini-project is to curate a diverse single-word answer VQA dataset using this multimodal data, evaluate pre-trained baseline models on it, and enhance performance through parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA). 
# Contents
- vrproject2baselineblip.ipynb - BLIP baseline
- vrproject2baselinevilt.ipynb - ViLT baseline
- convert.py - converts the llm outputs to csv format
- data_curation.ipynb - data creation 
- fine-tune-model.ipynb - LoRA fine-tuned BLiP
- vqa_generator.ipynb - vqa promt creation
