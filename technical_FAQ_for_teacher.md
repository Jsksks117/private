# Technical Q&A: Why We Chose Our Tech Stack
*Use this cheat sheet to confidently answer questions from your professor/panel regarding the engineering decisions behind your architecture.*

---

### 1. "Why did you use CTGAN instead of a standard GAN (like Vanilla GAN or WGAN)?"
* **The Problem with Standard GANs:** Standard GANs were designed for continuous, uniform data like images (pixels). If you feed a standard GAN "tabular data" (like your aviation CSVs), it fails miserably. Tabular data has *discrete categories* (e.g., Male/Female, Airline Categories) and *multi-modal continuous variables* (highly skewed numeric curves). 
* **The CTGAN Advantage:** **Conditional Tabular GAN (CTGAN)** specifically solves this. It uses **Mode-Specific Normalization** to tackle non-Gaussian, highly skewed numeric data, and uses a **Conditional Generator** to prevent "mode collapse" on extremely imbalanced categorical columns.

### 2. "Why use NetworkX to map the schema? Why not just generate each table independently?"
* **The Problem with Isolation:** If you generate the `Flights` table and `Passengers` table independently using two separate AI models, the relational integrity breaks. You will generate `Passengers` that belong to `flight_id`s that *don't exist* in the synthetic Flights table (these are called "orphaned records").
* **The NetworkX Advantage:** By representing the database as a **Directed Acyclic Graph (DAG)**, we mathematically calculate a **Topological Sort**. This guarantees that the AI *must* finish generating the Parent Table (Flights) first, and then conditionally feed those synthetic Primary Keys into the Child Table (Passengers) during generation, perfectly preserving Foreign Key referential integrity.

### 3. "Why use a 'Logarithmic Fan-out' for privacy budgeting? Why not split Epsilon (ε) equally across tables?"
* **The Problem with Equal Splitting:** If you give `Flights` and `Passengers` equal privacy budgets (say, ε=2.5 each), the noise added to the `Flights` table will corrupt its data. Because `Passengers` depend on `Flights`, that corruption cascades downstream, ruining the entire database.
* **Our Strategic Advantage:** We allocate **more** privacy budget (meaning *less* noise) to root/parent tables that have a high "fan-out" rate (many child dependencies). We use a log-normalization formula to ensure the Parent table is highly stable, which acts as a clean anchor preventing synthetic correlation errors from multiplying down the DAG.

### 4. "How is privacy enforced? Why use Meta's Opacus (DP-SGD) instead of just adding noise at the end?"
* **The Problem with Output Perturbation:** Classic data anonymization (like aggregating data and adding Laplace noise at the end) breaks the intricate machine-learning correlations (utility) of the data and leaves it vulnerable to modern hacker attacks.
* **The Opacus Advantage:** We use **Differentially Private Stochastic Gradient Descent (DP-SGD)**. Instead of adding noise to the data, Opacus adds noise to the AI's *learning process*. During backpropagation, it explicitly tracks per-sample gradients, strictly clips them to an L2 norm bound (preventing outsized influence from any single row), and injects calibrated Gaussian noise. This ensures the GAN mathematically cannot memorize a specific person's record.

### 5. "Why use Google dp-accounting (PRV Accountant) instead of Moments Accountant?"
* **The Advantage:** Tracking Privacy Loss (Epsilon drift) across hundreds of GAN epochs is computationally difficult. Older methods (like the Moments Accountant) provide "loose" mathematical boundaries, forcing you to stop training prematurely. Google's **PRV (Privacy Random Variables) Accountant** provides exact, tight boundaries for Poisson-sampled DP-SGD. This allows our GAN to train for more epochs (getting smarter and generating better data) while retaining the exact same mathematical DP guarantee.

### 6. "Why use DuckDB for Relational Evaluation instead of Pandas?"
* **The Advantage:** Pandas is extremely slow and memory-inefficient when it comes to joining large, multi-relational tables to check for Foreign Key compliance. **DuckDB** is an embedded, columnar analytical SQL engine. We use it to effortlessly cross-join our synthetic parent and child tables in a split second to mathematically prove our cardinality ratios and Orphan violation rates.

### 7. "What is TSTR, and why use it to evaluate the data instead of just looking at correlation?"
* **The Advantage:** Looking at a basic static correlation matrix doesn't prove the data is useful for Artificial Intelligence. We deploy a **Train on Synthetic, Test on Real (TSTR)** pipeline. We train a `scikit-learn` Random Forest Classifier entirely on our fake synthetic data. We then evaluate its predictive Accuracy and AUC against the isolated *real* data. If the model scores well, it is definitive proof that our synthetic data captured the true, complex multivariate signals of reality without copying it.
