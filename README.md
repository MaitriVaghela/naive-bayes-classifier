# Naive Bayes Classifier

A Python implementation of the Naive Bayes text classification algorithm, applied to multi-class newsgroup document classification using the **20 Newsgroups dataset**. Includes confusion matrix generation and accuracy visualization.

---

## Overview

This project implements a Naive Bayes classifier from scratch in Python to categorize newsgroup posts into predefined topic categories. Two classification setups are explored:

- **4-category classification** — e.g., `alt.atheism`, `talk.religion.misc`, `comp.graphics`, `sci.space`
- **20-category classification** — the full 20 Newsgroups dataset

The classifier uses Laplace (additive) smoothing with a configurable alpha parameter to handle unseen words gracefully.

---

## Repository Structure

```
Naive-Bayes-classifier/
├── Naive Bayes Classfier     # Core classifier implementation
├── Plotting and Accuracy     # Confusion matrix visualization and accuracy analysis
└── README.md
```

---

## How It Works

The Naive Bayes classifier applies Bayes' theorem with the "naive" assumption that features (words) are conditionally independent given the class label:

```
P(class | document) ∝ P(class) × ∏ P(word | class)
```

**Key steps:**
1. Tokenize and preprocess the input documents
2. Compute prior probabilities for each class
3. Compute word likelihoods per class using Laplace smoothing (α = 0.01)
4. Classify new documents by selecting the class with the highest posterior probability

---

## Dataset

The project uses the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/), a classic benchmark for text classification containing roughly 20,000 newsgroup documents across 20 topics, including:

| Group | Topics |
|-------|--------|
| Computers | `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x` |
| Recreation | `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey` |
| Science | `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space` |
| Politics & Religion | `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`, `alt.atheism`, `soc.religion.christian` |
| Miscellaneous | `misc.forsale` |

---

## Visualization

The `Plotting and Accuracy` script generates bar charts of the diagonal values from the confusion matrix (i.e., correctly classified samples per category) for both the 4-category and 20-category setups.

**Sample confusion matrix diagonal (4 categories, α = 0.01):**

| Category | Correctly Classified |
|----------|---------------------|
| alt.atheism | 66 |
| talk.religion.misc | 227 |
| comp.graphics | 380 |
| sci.space | 9 |

Charts are saved as:
- `Confusion Matrix for 4 categories.png`
- `Confusion Matrix for 20 categories.png`

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.01` | Laplace smoothing factor |
| `categories` | All 20 | Subset of newsgroup categories to classify |

To experiment with different smoothing values or category subsets, modify these parameters at the top of the classifier script.

---

## Results

With **α = 0.01**, the classifier achieves reasonable accuracy on well-separated topic categories (e.g., `comp.graphics`, `sci.space`) but struggles with thematically similar categories (e.g., politics and religion groups), which frequently get misclassified into `talk.politics.misc`.

---

