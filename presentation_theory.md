# Presenter's Oral Script & Theory Guide

*Use this document as your spoken script while advancing through the `demo_presentation.py` output or navigating the web frontend dashboard.*

---

## 🔹 Introduction

**"Good morning/afternoon, teachers and panel.** Today, my project addresses a massive challenge in modern data science: **Privacy vs. Utility.** Organizations like airlines and hospitals possess vast amounts of sensitive, highly relational data. However, due to strict privacy regulations, they cannot freely share it with researchers or external developers.

Our solution is a **Differentially Private Multi-Relational Synthetic Data Generator.** It utilizes advanced Generative Adversarial Networks—specifically CTGANs—to create synthetic copies of data that look and act exactly like the real thing natively mimicking the statistical properties, but securely guaranteeing without leaking a single real individual's private information."

---

## 🔹 Step 1: Mapping the Multi-Relational Schema 
*(Run script step 1 or point to the dashboard UI schema)*

**"First, we must understand the structure of the data.** 
We are working with an aviation database containing two main tables: **Flights** and **Passengers**. 
Passengers depend directly on Flights through a Foreign Key. We cannot train independent AI models on these tables in isolation because we would break the relational integrity of the database. 

To solve this, our system uses **NetworkX** to map the entire database schema into a Directed Acyclic Graph (DAG) logic. This automatically determines the correct chronological generation order—insisting that root tables (like Flights) are fully synthesized before simulating their dependent child tables (like Passengers)."

---

## 🔹 Step 2: Privacy Budget (Epsilon) Allocation
*(Run script step 2)*

**"This leads us to our Privacy framework.** 
Differential Privacy is strictly governed by a mathematical budget known as **Epsilon (ε).** Lower Epsilon means higher security, but more random noise is injected during AI training.

Because errors in our root table (Flights) will heavily cascade into the dependent table (Passengers), we use a proprietary **downstream fan-out algorithm**. The system parses the dependency graph and logarithmically distributes the global Epsilon Budget, assigning *more* Privacy Budget to the delicate upstream parent tables to ensure maximum relational stability down the line."

---

## 🔹 Step 3: Generative AI Processing (DP-CTGAN)
*(Run script step 3)*

**"Now the AI generates the data.** 
Our system natively spins up specialized Neural Networks known as **Conditional Tabular GANs** implemented in PyTorch. 

To guarantee perfect Differential Privacy during training, we integrated Meta's **Opacus** framework. Whenever the neural network updates its gradients to learn from confidential data, Opacus steps in. It calculates the *per-sample gradients*, explicitly clips them to a maximum threshold, and aggressively injects calibrated Gaussian Noise. 

Simultaneously, we track the privacy drift using **Google’s DP-Accounting Engine**. The microsecond our specified Privacy Budget is exhausted, the model forcefully stops training—guaranteeing that no specific raw data row was ever memorized or overfitted."

---

## 🔹 Step 4: Empirical Evaluation - Proving Security & Utility
*(Run script step 4)*

**"Finally, we must mathematically prove that the data holds its value and is truly secure.** Our evaluator audits the synthetic database across four vectors:

1. **Marginal Fidelity:** Using Kolmogorov-Smirnov statistical tests, we verify the underlying data distributions (such as average flight delays) are preserved.
2. **Relational Integrity:** Using DuckDB's in-memory SQL engine, we verify that Foreign Key dependencies weren't broken.
3. **Machine Learning Utility (TSTR):** We train a Random Forest predictive model purely on the synthetic data, and test it against the real data. Our accuracy proves the synthetic relationships holds high analytical power.
4. **Adversarial Privacy Audit:** Most importantly, we stress-test our data by simulating a hacker launching a **Membership Inference Attack (MIA)**. Because we enforced strict Differential Privacy, the attack completely fails to reverse-engineer matching real records, ensuring 100% data safety."

---

## 🔹 Conclusion
**"In conclusion,** this architecture provides a seamless, verifiable way to unlock the world's most confidential, interconnected relational databases safely, bridging the gap between rigorous data sharing and absolute computational privacy."
