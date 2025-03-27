## SmartTestAuto AI-Powered Test Failure Prediction

### Motivation for Choosing the Subject

---

Being a tester or QA, I’ve always been enthusiastic about getting better in my field and learning everything I can. I have been learning new things every day and every day has truly felt like a new beginning.

It all started when I chose to write my thesis on **test automation** and **machine learning integration**. It was quite a success. I did deep research into how machine learning can contribute to test automation and optimize the work of QA engineers.

That was the first clear signal leading me into machine learning. This thesis shaped my mindset and right after the examinations, I was thirsty for more knowledge in the AI field. I looked around and eventually earned a seat in the **AI Development program at ITH Vocational College** under the brilliant consult Robin Kamo.

That’s where this vision started to turn into reality.

## Project Origin

For the first assignment, we were asked to create a basic HTML front and implement version control. I wanted to build something meaningful from the start so I named my app:

> **SmartTestAuto**

Where the word “Smart” reflects the integration of machine learning later in the course.

---

### Choosing a Machine Learning Feature + Dataset

Since all my professional work revolves around **testing and QA**, it was a natural decision to build a machine learning feature focused on this domain. I chose a task that helps testers and developers predict whether a test might pass or fail based on historical data.

### Dataset Challenge

---

This was a _critical_ part as the dataset forms the ground on which the model stands.

It took me around two weeks to:

- Find legitimate public data
- Turn raw CSVs into usable, scaled, and normalized format
- Structure it for model training

I explored multiple sources and finally discovered the **Microsoft Azure Public Dataset** — which was a breakthrough. However, processing this raw data posed a new challenge.

### Understanding, Normalizing & Scaling the Dataset

---

The dataset was large and complex. Labels included:

- `value`
- `runtime`
- `starttime`
- `VM_id`

### Initial Strategy

---

At first, I thought longer runtime meant test failure — but deeper analysis revealed that:

- Some **long-runtime tests still had strong performance values**
- Some **short-runtime tests had poor values**

This observation varied across different test types.

So, I shifted my strategy:

> I focused on the **`value`** field as the most reliable indicator of pass/fail patterns.

### 🧪 Core Research Questions:

---

1. What does the dataset contain?
2. How can we define a test failure?
3. What patterns differentiate a failing test from a passing test?

I based the hypothesis on this:

> **High runtime + Low performance = Possible failure**

Example source:

```
test_suite=perf-bench/test_name=Benchmark:_Syscall_Basic/
vm_lifespan=long/vm_region=westus2/vm_sku=B8ms/unit=ops_sec.csv
```

---

### Data Extraction & Cleaning

---

Initially, extracting all CSVs into one root directory failed — due to duplicate filenames (`unit=ops_sec.csv`). I solved this by:

- Pulling selected CSVs into a new directory
- Renaming based on folder path context

This process gave me:

- 15 unique CSVs
- Each with over 10,000 rows
- Total of **67,099 rows** after merge. A huge milestone!

### Extracting Failing Tests

---

To isolate failures, I targeted the **bottom 10% of performance `value` scores** across all datasets.

I created a script to extract and label this subset — saved into:

```
STA-DataSet-Low10/
```

Then, after confirming with my teacher, I merged all subsets into one unified dataset — using an iterative merging strategy to avoid data corruption due to size.

### 🧮 Normalization with Z-Score

---

Before training, I chose Z-Score Normalization over Min-Max Scaling, because:

- It handles **outliers** better
- It preserves **relative distance** between values

### Formula Used:

```
Z = (X - μ) / σ
X = Original value
μ = Mean
σ = Standard deviation
```

### Choosing the Right Algorithm

---

Since this was a classification task (Pass = 0, Fail = 1), I tested two algorithms:

- **Logistic Regression**
- **Random Forest**

But first, I normalized the input features: `value` and `runtime`.

---

### Training the Model: Logistic Regression

---

File: `train-logistic.py`

Steps:

1. Load dataset
2. Create binary label:
   ```python
   is_failed = 1 if value < 0 else 0
   ```
3. Drop non-numeric fields: `starttime`, `sourcefile`, etc.
4. Use:
   ```python
   X = input features
   y = is_failed
   ```
5. Split into 80% train / 20% test with `random_state=42` for reproducibility
6. Train model with:
   ```python
   LogisticRegression(max_iter=1000)
   ```

### Logistic Regression Coefficient & Sigmoid

---

The coefficient value was **-42.98**, calculated from the dataset automatically.

The sigmoid formula:

```
sigmoid(value × coef + bias) ≈ is_failed
```

I validated this with a histogram that showed values clustered around 0, explaining the model's large coefficient weight.

---

### 🧪 Evaluation Metrics:

```
Accuracy: 0.99
Precision: 1.00
Recall:    0.97
F1 Score:  0.99
```

### Confusion Matrix:

|             | Predicted Pass (0) | Predicted Fail (1) |
| ----------- | ------------------ | ------------------ |
| Actual Pass | ✅ 9397            | ❌ 0               |
| Actual Fail | ❌ 108             | ✅ 3915            |

✅ The model missed only 108 out of 4023 actual failures — a strong result.

### Random Forest Classifier

---

File: `train-randomforest.py`

Out of curiosity, I trained a second model — and was amazed:

### Evaluation Metrics:

```
Accuracy: 1.00
Precision: 1.00
Recall:    1.00
F1 Score:  1.00
```

A perfect score across the board.

It was truly mind-blowing to see how well Random Forest performed on the same dataset. It shows that the dataset was strong, the preprocessing was effective, and the patterns were clear for multiple classifiers.

## Credits

Thanks to **Robin Kamo**, my teacher at ITH, for brilliant guidance and for turning advanced concepts into something accessible and inspiring.
