# viterbi_vs_greedy

This repository compares the performance of the Viterbi algorithm with a greedy search approach for a sequence labeling task, such as part-of-speech tagging. The key functions within the main file allow for testing and evaluating different approaches on a given dataset.

## To Run:

### 1. Running Experiments with Different Alpha Values:

You can experiment with different values of alpha to see how they affect performance. By default, alpha = 1 is the best-performing value. To test other values of alpha, uncomment the following line in the `main()` function:

```python
# run_experiments()
```

### 2. Running the Greedy Search Model:

To test the greedy search model on the test set, uncomment the following line in the `main()` function:

```python
# run_greedy()
```

This will run the greedy search method to predict the tags for the test set and display the evaluation results.

### 3. Running the Baseline Model:

To test the baseline model, which only uses emission probabilities (without transition probabilities), uncomment the following line in the `main()` function:

```python
# run_baseline()
```

This will run the baseline model, which is expected to perform worse compared to models that incorporate transition probabilities.

### 4. Running the Viterbi Algorithm on the Test Set:

The Viterbi algorithm is slower than the greedy search but provides more accurate results in terms of overall accuracy. To get the results for Viterbi, uncomment the following line:

```python
# run_test()
```

**Note**: Running this function takes a long time due to the complexity of the Viterbi algorithm.

Additionally, all predictions will be saved as a JSON file during this process.

### 5. Evaluating Viterbi Results:

Once you have the Viterbi results in a JSON file, you can evaluate them in comparison to the test data by uncommenting the following line:

```python
# evaluate_viterbi_test()
```

This will load the JSON file, compare the predictions to the actual test data, and output the evaluation metrics.

## NOTE: you will also be required to change all filepaths to those on your local computer
---

## Model Evaluation Metrics:

The models are evaluated using the following metrics:
- Precision
- Recall
- F1-Score
- Accuracy

These metrics are displayed in a confusion matrix and are printed to the console after each evaluation.

---

## Performance Notes:

- **Greedy Search**: The greedy search model performs relatively well with high precision, recall, and F1-scores. However, it does not account for the entire sequence of tokens and can make independent decisions for each token. This approach is faster but may miss context-dependent patterns.
  
- **Viterbi Algorithm**: The Viterbi algorithm performs better in terms of accuracy by considering the entire sequence of tags. It uses transition probabilities to account for dependencies between tags, leading to a more globally optimal sequence of tags.

- **Baseline**: The baseline model, which only uses emission probabilities, is expected to underperform compared to both greedy and Viterbi models, as it ignores transition probabilities between tags.

---

## Requirements:

- Python 3.x
- Libraries: `numpy`, `pandas`, `sklearn`, `matplotlib`, `seaborn`

You can install the required dependencies using the following command:

pip install -r requirements.txt

---

## Conclusion:

This repository demonstrates the trade-offs between speed and accuracy in sequence labeling tasks. The Viterbi algorithm, while slower, yields more accurate results by considering the full context of the sequence. The greedy search, on the other hand, offers a faster alternative but with potentially lower performance.

