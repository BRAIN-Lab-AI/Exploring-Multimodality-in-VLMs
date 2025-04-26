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

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to Saad Alghamdi, Abdulaziz Alshukri and Mohammed Alharthi for the amazing team effort, invaluable guidance and support throughout this project.
