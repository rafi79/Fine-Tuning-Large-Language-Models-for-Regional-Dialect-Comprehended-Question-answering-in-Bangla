
🧠 Fine-Tuning Large Language Models for Regional Dialect Comprehended Question Answering in Bangla
📄 Citation
Md Jahid Alam Riad, Prosenjit Roy, Mahfuzur Rahman Shuvo, Nobanul Hasan, Stabak Das, Fateha Jannat Ayrin, Syeda Sadia Alam, Afsana Khan, Md Tanzim Reza, Md Mizanur Rahman,
"Fine-Tuning Large Language Models for Regional Dialect Comprehended Question Answering in Bangla,"
2025 IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS), Bhopal, India, 2025, pp. 1–6.
DOI: 10.1109/SCEECS64059.2025.10940303

📚 Abstract
For diverse languages like Bangla, maintaining regional dialects poses a major challenge. Dialects from regions such as Chittagong, Noakhali, Sylhet, Barishal, and Mymensingh can be mutually unintelligible, hindering smooth communication.

This study introduces a new dialect-rich dataset of 12,500 sentence pairs, each reflecting region-specific phrasing and their answers in the same dialect. We then applied Low-Rank Adaptation (LoRA) to fine-tune four large language models for dialect-aware question answering.

🧪 Key Findings
ChatGPT-4o: BLEU score 53%

Claude 3.5 Sonnet: 46%

Gemma-2-9B: 42%

Mistral-7B: 40%

The results show that large language models exhibit significant variance in regional dialect understanding, with ChatGPT-4o outperforming others across all metrics.

📌 Keywords
sql
Computer science, Adaptation models, Large language models, Chatbots, 
Question answering, Bangla, Dialect, ChatGPT, Claude
🏆 Contributions
✅ Creation of a comprehensive dialect dataset of Bangla covering five major regions
✅ Demonstrated the efficacy of LoRA for efficient fine-tuning of LLMs
✅ Comparative evaluation of four state-of-the-art models
✅ Pioneered the study of regional dialect handling in Bangla QA systems

📊 Dataset Details
Size: 12,500 sentence-answer pairs
Paper Dataset Link-https://www.kaggle.com/datasets/mizanurrahmanrafi/bangliandialectllm-qa

Dialects: Chittagong, Noakhali, Sylhet, Barishal, Mymensingh

Structure: JSON/CSV format with dialect tagging

(Availability depends on licensing and publication guidelines – not publicly linked here)

⚙️ Model Fine-Tuning Approach
Model	Fine-Tuning Technique	Adapter Used	BLEU Score
ChatGPT-4o	LoRA + Prompt-tuning	Proprietary	53%
Claude 3.5 Sonnet	LoRA + Prompt-tuning	Proprietary	46%
Gemma-2-9B	LoRA (PEFT)	PEFT	42%
Mistral-7B	LoRA (PEFT)	PEFT	40%

📅 Conference Information
Event: IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS)

Date: 18–19 January 2025

Location: Bhopal, India

Publisher: IEEE

DOI: 10.1109/SCEECS64059.2025.10940303


Expand dataset to cover more dialects and code-switching use cases

Investigate multimodal QA with audio/text input for spoken dialect comprehension

Explore distillation of dialect-aware models for deployment on low-resource devices

🙏 Acknowledgements
We express gratitude to the contributing institutions and volunteers who supported data collection in regional dialects. Special thanks to IEEE SCEECS 2025 organizers for providing a platform for this research.
