# Fine-Tuning Large Language Models for Regional Dialect Comprehended Question Answering in Bangla

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FSCEECS64059.2025.10940303-blue)](https://doi.org/10.1109/SCEECS64059.2025.10940303)
[![Conference](https://img.shields.io/badge/Conference-IEEE%20SCEECS%202025-green)](https://sceecs.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/datasets/mizanurrahmanrafi/bangliandialectllm-qa)

## 📋 Overview

This repository contains the implementation and resources for fine-tuning Large Language Models (LLMs) to understand and respond in regional Bangla dialects. Our work addresses the significant challenge of maintaining regional dialect comprehension in diverse languages like Bangla, where dialects from different regions can be mutually unintelligible.

## 🎯 Problem Statement

Regional dialects in Bangla from areas such as Chittagong, Noakhali, Sylhet, Barishal, and Mymensingh pose major communication barriers. Existing language models struggle to understand and respond appropriately to these dialect variations, limiting their effectiveness for native speakers of these regional variants.

## ✨ Key Contributions

- **Comprehensive Dataset Creation**: Built a dialect-rich dataset of 12,500 sentence pairs covering five major Bangla regional dialects
- **LoRA Fine-tuning Methodology**: Demonstrated the efficacy of Low-Rank Adaptation (LoRA) for efficient fine-tuning of LLMs
- **Comparative Model Evaluation**: Benchmarked four state-of-the-art language models on dialect-aware question answering
- **Pioneer Research**: First comprehensive study of regional dialect handling in Bangla QA systems

## 📊 Results Summary

| Model | Fine-Tuning Technique | BLEU Score | Performance Rank |
|-------|----------------------|------------|------------------|
| **ChatGPT-4o** | LoRA + Prompt-tuning | **53%** | 🥇 1st |
| **Claude 3.5 Sonnet** | LoRA + Prompt-tuning | **46%** | 🥈 2nd |
| **Gemma-2-9B** | LoRA (PEFT) | **42%** | 🥉 3rd |
| **Mistral-7B** | LoRA (PEFT) | **40%** | 4th |

## 🗂️ Dataset Information

### Dataset Specifications
- **Size**: 12,500 sentence-answer pairs
- **Format**: JSON/CSV with dialect tagging
- **Coverage**: Five major Bangla regional dialects
- **Structure**: Question-answer pairs with regional dialect variations

### Covered Dialects
1. **Chittagong** (চট্টগ্রামের উপভাষা)
2. **Noakhali** (নোয়াখালীর উপভাষা)
3. **Sylhet** (সিলেটের উপভাষা)
4. **Barishal** (বরিশালের উপভাষা)
5. **Mymensingh** (ময়মনসিংহের উপভাষা)

### Dataset Access
📥 **Download**: [Bangla Dialect LLM-QA Dataset](https://www.kaggle.com/datasets/mizanurrahmanrafi/bangliandialectllm-qa)

*Note: Dataset availability depends on licensing and publication guidelines*

## 🔧 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/bangla-dialect-llm-finetuning.git
cd bangla-dialect-llm-finetuning

# Install dependencies
pip install -r requirements.txt

# Install additional packages for LoRA fine-tuning
pip install peft transformers accelerate datasets
```

## 🚀 Usage

### 1. Data Preparation
```python
from src.data_loader import BanglaDialectDataset

# Load the dialect dataset
dataset = BanglaDialectDataset("path/to/dataset.csv")
train_data, val_data, test_data = dataset.split_data()
```

### 2. Model Fine-tuning
```python
from src.fine_tuning import LoRAFineTuner

# Initialize fine-tuner for Gemma-2-9B
fine_tuner = LoRAFineTuner(
    model_name="google/gemma-2-9b",
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1
    }
)

# Fine-tune the model
fine_tuner.train(train_data, val_data, epochs=3)
```

### 3. Evaluation
```python
from src.evaluation import BLEUEvaluator

# Evaluate model performance
evaluator = BLEUEvaluator()
bleu_score = evaluator.evaluate(model, test_data)
print(f"BLEU Score: {bleu_score:.2f}%")
```

## 📁 Project Structure

```
bangla-dialect-llm-finetuning/
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data_loader.py          # Data loading utilities
│   ├── fine_tuning.py          # LoRA fine-tuning implementation
│   ├── evaluation.py           # Model evaluation metrics
│   └── utils.py                # Helper functions
├── models/
│   ├── checkpoints/            # Model checkpoints
│   └── configs/                # Model configurations
├── notebooks/
│   ├── data_analysis.ipynb     # Dataset exploration
│   ├── model_comparison.ipynb  # Model performance analysis
│   └── dialect_examples.ipynb  # Dialect variation examples
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔬 Methodology

### Fine-tuning Approach
We employed **Low-Rank Adaptation (LoRA)** technique for efficient fine-tuning:

- **Parameter Efficiency**: LoRA adds trainable low-rank matrices to existing model layers
- **Memory Optimization**: Significantly reduces memory requirements compared to full fine-tuning
- **Preservation**: Maintains pre-trained knowledge while adapting to dialect-specific patterns

### Evaluation Metrics
- **BLEU Score**: Primary metric for measuring translation quality
- **Dialect Accuracy**: Correctness of dialect-specific responses
- **Semantic Similarity**: Preservation of meaning across dialect variations

## 📈 Future Work

- [ ] **Dataset Expansion**: Include more regional dialects and code-switching scenarios
- [ ] **Multimodal Integration**: Investigate audio/text input for spoken dialect comprehension
- [ ] **Model Distillation**: Explore lightweight models for deployment on resource-constrained devices
- [ ] **Real-time Applications**: Develop chatbots and voice assistants with dialect awareness

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{riad2025finetuning,
  title={Fine-Tuning Large Language Models for Regional Dialect Comprehended Question Answering in Bangla},
  author={Riad, Md Jahid Alam and Roy, Prosenjit and Shuvo, Mahfuzur Rahman and Hasan, Nobanul and Das, Stabak and Ayrin, Fateha Jannat and Alam, Syeda Sadia and Khan, Afsana and Reza, Md Tanzim and Rahman, Md Mizanur},
  booktitle={2025 IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS)},
  pages={1--6},
  year={2025},
  organization={IEEE},
  address={Bhopal, India},
  doi={10.1109/SCEECS64059.2025.10940303}
}
```

## 👥 Authors

- **Md Jahid Alam Riad** - Lead Researcher
- **Prosenjit Roy** - Co-researcher
- **Mahfuzur Rahman Shuvo** - Data Engineer
- **Nobanul Hasan** - Model Developer
- **Stabak Das** - Evaluation Specialist
- **Fateha Jannat Ayrin** - Dataset Curator
- **Syeda Sadia Alam** - Research Assistant
- **Afsana Khan** - Research Assistant
- **Md Tanzim Reza** - Technical Lead
- **Md Mizanur Rahman** - Project Supervisor

## 🏢 Conference Information

**IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS) 2025**
- **Date**: January 18-19, 2025
- **Location**: Bhopal, India
- **Publisher**: IEEE
- **DOI**: [10.1109/SCEECS64059.2025.10940303](https://doi.org/10.1109/SCEECS64059.2025.10940303)


## 🙏 Acknowledgments

We express our sincere gratitude to:
- Contributing institutions and volunteers who supported regional dialect data collection
- IEEE SCEECS 2025 organizers for providing a platform for this research
- The open-source community for tools and libraries that made this work possible

## 📧 Contact

For questions or collaborations, please reach out to:
- **Primary Contact**: jahidalamriad@gmail.com
- **Project Lead**: rafi79466@gmail.com
- **Issues**: Use GitHub Issues for technical questions

---

**Keywords**: Computer science, Adaptation models, Large language models, Chatbots, Question answering, Bangla, Dialect, ChatGPT, Claude
