# MASC 515 Assignment 3 – microGPT Enhancements

## Overview

This project extends the original **microGPT** implementation by integrating several modern techniques used in transformer-based models. The goal is to demonstrate understanding and implementation of advanced neural network components in a minimal Python-based GPT model.

The following algorithms were implemented:

* Gaussian Error Linear Units (GELU)
* Low-Rank Adaptation (LoRA)
* Rotary Positional Embeddings (RoPE)
* Mixture of Experts (MoE)

---

## 1. Gaussian Error Linear Units (GELU)

### Idea

GELU is an activation function commonly used in transformer models (e.g., BERT, GPT). Unlike ReLU, which simply clips negative values, GELU provides a smooth, probabilistic activation.

It can be interpreted as:

> Scaling inputs by their probability under a Gaussian distribution.

### Why it is useful

* Smooth and differentiable everywhere
* Improves training stability
* Empirically performs better than ReLU in transformers

### Implementation

The ReLU activation in the MLP block was replaced with GELU using an approximate formulation based on the tanh function.

---

## 2. Low-Rank Adaptation (LoRA)

### Idea

LoRA reduces the cost of training large models by avoiding full weight updates. Instead of modifying a full weight matrix, it adds a **low-rank decomposition**:

W' = W + A × B

Where:

* W = original weight
* A, B = small low-rank matrices

### Why it is useful

* Reduces number of trainable parameters
* More memory-efficient
* Enables fast fine-tuning

### Implementation

LoRA was applied to the query projection in the attention mechanism:

* A low-rank update is computed
* Added to the original linear projection output

---

## 3. Rotary Positional Embeddings (RoPE)

### Idea

RoPE encodes positional information by **rotating embeddings** in vector space rather than adding position vectors.

Instead of:
x = token_embedding + position_embedding

RoPE applies a rotation:

* Uses sine and cosine functions
* Rotates pairs of embedding dimensions

### Why it is useful

* Captures relative positions naturally
* Generalizes better to longer sequences
* Used in modern LLMs (e.g., LLaMA, GPT-NeoX)

### Implementation

* Removed learned positional embeddings
* Applied rotation to token embeddings based on position index

---

## 4. Mixture of Experts (MoE)

### Idea

MoE replaces a single feedforward network with multiple “experts” and a gating function that selects how much each expert contributes.

Instead of one MLP:

* Multiple MLPs (experts) are used
* A gating network computes weights

Final output:
Weighted combination of expert outputs

### Why it is useful

* Increases model capacity without large computation cost
* Only a subset of experts is effectively used per input
* Scales efficiently

### Implementation

* Replaced the original MLP block with multiple expert networks
* Added a gating layer to compute probabilities
* Final output is a weighted sum of expert outputs

---

## Conclusion

This project demonstrates how modern transformer improvements can be incorporated into a minimal GPT implementation. Each method improves different aspects of the model:

| Technique | Benefit                      |
| --------- | ---------------------------- |
| GELU      | Better activation function   |
| LoRA      | Efficient parameter updates  |
| RoPE      | Improved positional encoding |
| MoE       | Scalable model capacity      |

These enhancements reflect techniques used in state-of-the-art large language models.

---

## References

* GELU: https://arxiv.org/abs/1606.08415
* LoRA: https://arxiv.org/abs/2106.09685
* RoPE: https://arxiv.org/abs/2104.09864
* MoE: https://huggingface.co/blog/moe#a-brief-history-of-moes

---
