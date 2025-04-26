# EXPLORE: Examining the Potential of Multimodality for Vision-Language Models


## Introduction
This project explores advanced techniques in multimodal AI by enhancing the InstructBLIP vision-language model—a state-of-the-art architecture designed to interpret visual content through instruction-guided prompting. InstructBLIP leverages a powerful query transformer to conditionally extract image features based on natural language instructions, enabling robust zero-shot performance across a wide range of tasks such as image captioning, visual question answering, and visual search. To further optimize the model for deployment in resource-constrained environments, we integrate Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning strategy that significantly reduces memory and computational overhead. By systematically evaluating multiple architectural configurations and fine-tuning methods, this work aims to maintain at least 70% of the original model's performance while drastically reducing its complexity—paving the way for scalable, real-world applications in intelligent visual understanding.

## Project Metadata
### Authors
- **Team:** Mohammed Alharhti, Abdulaziz Alshukri and Saad Alghamdi
- **Supervisor Name:** Dr. Muzammil Behzad

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500)

### Reference Dataset
- [VL-RewardBench Dataset](https://huggingface.co/datasets/MMInstruction/VL-RewardBench)


## Project Technicalities

### Terminologies
- **Vision-Language Model:** A model designed to jointly process and understand visual and textual inputs for tasks such as image captioning or visual question answering.
- **InstructBLIP:** An instruction-tuned vision-language model that leverages query-based prompting to enhance image-text understanding in zero-shot and few-shot settings.
- **Query Transforme:** A transformer architecture variant that conditions image features based on textual prompts, allowing for instruction-specific representation learning.
- **Zero-Shot Learning:** A setting where a model performs tasks it hasn’t explicitly been trained on, using generalization learned from related data.
- **LoRA (Low-Rank Adaptation):** A fine-tuning technique that introduces low-rank matrices into transformer layers, significantly reducing the number of trainable parameters.
- **Parameter-Efficient Fine-Tuning:** A method of adapting pre-trained models using a small number of trainable parameters to save computational resources and memory.
- **Visual Question Answering (VQA):** A task where a model answers natural language questions about an image, requiring both visual understanding and language reasoning.
- **Model Compression:** Techniques aimed at reducing a model’s size and complexity while preserving its performance, facilitating deployment on edge devices.

### Proposed Solution: Code-Based Implementation
This repository contains notebook-based implementations for fine-tuning the InstructBLIP model using LoRA, with experiments on various architecture settings and performance metrics:

- **LoRA Integration:** LoRA applied to Q-Former and selected LLM layers.
- **Configurable Architecture:** Experimentation with hidden size, attention heads, and intermediate layers.
- **Multimodal Evaluation:** Evaluation using VL-RewardBench dataset and MME benchmark.

### Key Components
- **`InsructBlip_fine_tuned_Default.ipynb`**: Contains the modified InstructBLIP architecture with Lora.
- **`MME_BenchMark_InstructBlip_Vicuna7B_.ipynb`**: MME BenchMark to Evaluate the model.
- **`evaluation.ipynb`**: Evaluating the Model using ROUGE metrics for text quality and CLIP scores for visual-textual alignment.

## Model Workflow  
The workflow of the fine-tuned InstructBLIP model with LoRA is designed to generate natural language outputs (such as answers or captions) from image-text pairs through a multi-stage vision-language processing pipeline:

1. **Input:**
   - **Instruction Prompt + Image:** The model takes a text instruction and an image input (e.g., "Describe the scene in the image").
   - **Tokenization:** The instruction is processed into embeddings using a language encoder (such as T5 or similar).
   - **Image Encoding:** The image is encoded using a vision transformer or CLIP-like image encoder to obtain visual embeddings.

2. **Query Transformer (Q-Former):**
   - **Feature Extraction:** The Q-Former extracts relevant features from the image, guided by the instruction.
   - **Multimodal Embedding:** Outputs refined embeddings conditioned on both modalities (text and image).

3. **Language Model Decoding:**
   - **Text Generation:** The conditioned embeddings are passed through a language model decoder (e.g., T5) to generate natural language outputs such as answers or captions.

4. **Fine-Tuning with LoRA:**
   - **LoRA Integration:** LoRA layers are inserted into the Q-Former and LLM to reduce training parameters while maintaining performance.
   - **Training Dataset:** The model is fine-tuned using the VL-RewardBench dataset.

5. **Evaluation:**
   - **ROUGE Metrics:** ROUGE-1, ROUGE-2, and ROUGE-L are used to assess the quality of the generated text.
   - **CLIP Score:** Used to evaluate the alignment between visual and textual embeddings. Higher scores indicate better alignment.
   - **MME Benchmark:** The model is evaluated on the MME benchmark to assess its performance on multimodal vision-language tasks.

## How to Run the Code

This code can be run directly on **Google Colab** with an **A100 GPU** for optimal performance.

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to Saad Alghamdi, Abdulaziz Alshukri and Mohammed Alharthi for the amazing team effort, invaluable guidance and support throughout this project.
