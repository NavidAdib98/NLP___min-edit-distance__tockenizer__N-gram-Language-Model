# 🧠 NLP Fundamentals Project — Tokenization, Regular Expressions, and N-Gram Language Modeling  

## 📘 Project Overview  
In this project, we explore and implement several **core Natural Language Processing (NLP)** concepts — including **Regular Expressions**, **Tokenization**, and **N-Gram Language Modeling**.  
Through a series of practical exercises, each topic is implemented from scratch to gain a deeper understanding of the mathematical and computational foundations of language models.

---

## 📂 Table of Contents  
1. [Regular Expression & Minimum Edit Distance](#-regular-expression--minimum-edit-distance)  
2. [Tokenization](#-tokenization)  
3. [N-Gram Language Modeling](#-n-gram-language-modeling)  
4. [Results & Analysis](#-results--analysis)  
5. [References](#-references)

---

## 🔹 Regular Expression & Minimum Edit Distance  

### 🔸 Email Validation using Regex  
In this section, we used **advanced regular expressions** to validate complex email patterns.  
Features implemented:
- **Capturing groups**
- **Lookahead assertions**
- **Domain and subdomain validation**
- **Ensuring both first and last names appear in the email**

```python
pattern = re.compile(
    r'^'
    r'(?!.*\..)'                          
    r'name=([A-Za-z]+)\s+([A-Za-z]+),'
    r'\s*email='
    r'(?:(?=.*\1)(?=.*\2)[A-Za-z0-9._-]*)'
    r'@[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*'
    r'\.[A-Za-z]{2,}'
    r'$',
    re.IGNORECASE
)
```

### 🔸 Auto-Correction using Minimum Edit Distance  
The **Minimum Edit Distance (MED)** algorithm was implemented using **Dynamic Programming (DP)** to compute the minimum number of operations (insertions, deletions, substitutions) required to transform one word into another.  

**Mathematical definition:**
$$
D(i, j) =
\begin{cases}
0 & \text{if } i = 0 \text{ and } j = 0 \\
j & \text{if } i = 0 \\
i & \text{if } j = 0 \\
\min
\begin{cases}
D(i-1, j) + 1 \\
D(i, j-1) + 1 \\
D(i-1, j-1) + cost
\end{cases}
& \text{otherwise}
\end{cases}
$$

The final implementation was used for **word spelling correction** — comparing each word with a dictionary and suggesting the closest match.

---

## 🔹 Tokenization  

### 🔸 Rule-Based Tokenizer  
Implemented a **rule-based tokenizer** capable of handling:
- Abbreviations (e.g., "دکتر.")
- Acronyms
- Punctuation
- Numeric tokens  
This tokenizer used pattern-based rules for accurate word segmentation.

### 🔸 BPE and WordPiece Tokenizers  
Later, **Byte Pair Encoding (BPE)** and **WordPiece** tokenizers were trained on Persian text data using  
[`taesiri/TinyStories-Farsi`](https://huggingface.co/taesiri/TinyStories-Farsi) from Hugging Face.

These tokenizers segment words into subword units and improve handling of rare or unseen words.

---

## 🔹 N-Gram Language Modeling  

### 📘 Model Description  
The **NGramLanguageModel** class was implemented from scratch with full training, smoothing, and generation functionalities.

```python
class NGramLanguageModel:
    def __init__(self, n):
        ...
    def train(self, tokenized_sentences):
        ...
    def generate(self, max_tokens=100, smoothing="none"):
        ...
    def perplexity(self, tokenized_sentences, smoothing="none"):
        ...
```

Each model stores N-gram counts and context probabilities in memory (using dictionaries of `Counter` objects), which may grow substantially depending on corpus size.

---

### 🔸 Mathematical Formulation  

#### N-Gram Probability:
$$
P(w_t | w_{t-n+1}^{t-1}) = \frac{C(w_{t-n+1}^{t})}{C(w_{t-n+1}^{t-1})}
$$

#### Perplexity:
$$
PP(W) = \exp\left( -\frac{1}{N} \sum_{t=1}^{N} \log P(w_t | w_{t-n+1}^{t-1}) \right)
$$

---

### 🔸 Implemented Smoothing Methods  

1. **Laplace (Add-One) Smoothing:**
   $$
   P_{laplace}(w_i|h) = \frac{C(h, w_i) + 1}{C(h) + |V|}
   $$

2. **Linear Interpolation:**
   $$
   P_{interp}(w_i|h) = \sum_{j=1}^{n} \lambda_j P(w_i|h_j)
   $$

3. **Backoff:**
   $$
   P_{backoff}(w_i|h) =
   \begin{cases}
   P(w_i|h) & \text{if } C(h, w_i) > 0 \\
   \alpha \cdot P(w_i|h') & \text{otherwise}
   \end{cases}
   $$

---

### 🔸 Temperature-Based Sampling  

To simulate creativity and randomness during text generation, **temperature scaling** was applied:

$$
p_i^\text{new} = \frac{p_i^{1/T}}{\sum_j p_j^{1/T}}
$$

- **Low temperature (T < 1):** More predictable, repetitive sentences  
- **High temperature (T > 1):** More diverse but possibly incoherent text  

#### Example Results  

**2-Gram Model**
```
زمانی که شکر ، بنابراین رفتند . او دوست داشتند . او را که می‌دویدند ، یک سگ خیلی ترسیدند . ...
```

**8-Gram Model**
```
مامان و بابا توی آشپزخانه بودند . مامان گریه می‌کرد . " چه اتفاقی افتاد ؟" بابا پرسید . مامان یک تقویم را بالا گرفت و گفت ...
```

---

## 🔹 Perplexity Evaluation  

The **perplexity metric** was implemented to measure how well the model predicts unseen text.  
Each smoothing technique was tested, showing that:
- **Interpolation** achieved the most balanced results.  
- **Laplace** improved rare word handling but over-smoothed probabilities.  
- **Backoff** performed well for small datasets.

---

## 🔹 Results & Analysis  

| Method | Description | Observation |
|--------|--------------|--------------|
| **Low Temperature (0.5)** | Deterministic output | Grammatically correct but repetitive |
| **Medium Temperature (1.0)** | Balanced randomness | Fluent and creative text |
| **High Temperature (2.0)** | High diversity | Nonsensical or incoherent output |

As the **temperature increases**, text diversity improves but coherence decreases.

---

## 🔹 Tools and Environment  
- **Python 3.10**  
- **Google Colab** (for model training and experiments)  
- **Hugging Face Datasets** (Persian text corpus)  
- **NumPy**, **Regex**, **Matplotlib** (for analysis and visualization)

---

## 💡 Author Notes  

This project was implemented as part of an NLP course.  
Throughout the exercises:
- The core model (`NGramLanguageModel`) and its methods were gradually refined and rewritten to follow academic standards.  
- Model training and generation were performed in **Google Colab**, and all code and outputs were copied here for submission and version control.


