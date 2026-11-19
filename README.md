# Medical Radiology Report Generation

An end-to-end vision-language framework that generates clinical radiology reports from frontal chest X-ray images. The architecture pairs a domain-specific visual encoder (TorchXRayVision ResNet-50) with a biomedical causal language model (Microsoft BioGPT) via prefix-conditioned visual tokens.

## Architecture

- **Vision Backbone**: Pre-trained ResNet-50 from TorchXRayVision (`resnet50-res512-all`), extracting spatial feature maps of shape `[B, 2048, 16, 16]`.
- **Visual Projection**: Projects spatial features into 256 tokens of dimension 1024 (`[B, 256, 1024]`) using Linear projection, LayerNorm, GELU, and Dropout.
- **Language Model**: Microsoft BioGPT (`microsoft/biogpt`). The visual tokens are encoded and passed as prefix key-value states (`past_key_values`) to condition autoregressive text decoding.

## Dataset

- **Source**: Indiana University Chest X-Ray Collection.
- **Pre-processing**: Filtered for frontal views (AP/PA), merged findings and impression fields into clean report targets, and split 80% train / 20% validation.
- Images resized to 512x512 grayscale tensors.

## Evaluation Results

Evaluated on unseen validation chest X-rays:

| Metric | Score |
| --- | --- |
| BLEU-1 | 0.3679 |
| BLEU-2 | 0.1293 |
| BLEU-3 | 0.0513 |
| BLEU-4 | 0.0212 |
| ROUGE-1 | 0.3392 |
| ROUGE-2 | 0.1025 |
| ROUGE-L | 0.2251 |

## Repository Structure

```text
.
├── .gitignore               # Ignores large weight checkpoints and local caches
├── notebook-bio.ipynb       # Complete training, validation, and evaluation pipeline
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

*Note: Checkpoint weights (`medical_biogpt_weights.pt`, ~1.49 GB) are excluded from version control to comply with repository file size limits.*

## Getting Started

### Prerequisites

Python 3.10+ with CUDA-enabled PyTorch is recommended.

```bash
pip install -r requirements.txt
```

### Running the Notebook

Open and execute the pipeline in Jupyter Notebook or Kaggle:

```bash
jupyter notebook notebook-bio.ipynb
```

The notebook covers:
1. Data loading and text normalization.
2. PyTorch Dataset and custom collation with BioGPT tokenization.
3. Multimodal model definition (ResNet-50 + BioGPT prefix fusion).
4. Mixed-precision training with gradient accumulation.
5. Inference and automated evaluation (BLEU and ROUGE).
