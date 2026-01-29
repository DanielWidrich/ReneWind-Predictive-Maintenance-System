# ReneWind-Predictive-Maintenance-System
Failure Detection for Wind Turbine Operations

## Business Problem

Wind energy operators face significant costs from unexpected turbine failures, including equipment replacement, unplanned downtime, and lost energy production. Predictive maintenance aims to reduce these costs by identifying failure patterns early and enabling proactive repairs before breakdowns occur.

ReneWind collected sensor-based operational data from wind turbines to support a machine learning solution that can predict generator failures in advance. The objective of this project is to build and tune classification models that accurately detect impending failures while accounting for the real-world cost tradeoffs between inspections, repairs, and replacements.

---

## Forensic Research & Key Insights

Analysis of the dataset revealed several important operational considerations:

- **Failure detection is a cost-sensitive problem**, where missed failures (false negatives) are substantially more expensive than false alarms.
- Sensor data contains strong predictive signal, but patterns are distributed across many features, limiting interpretability.
- Overly simple models underperform, while carefully regularized neural networks generalize well to unseen data.
- Model evaluation must prioritize **recall on failure cases** rather than overall accuracy.

These findings guided model selection, metric prioritization, and threshold strategy.

---

## Modeling & Evaluation

Multiple models were evaluated, including neural network architectures with different depths and regularization strategies. The final selected model (Model 3: two hidden layers with dropout, optimized using SGD) achieved the strongest balance between failure detection and operational cost control:

- **Recall (Failure): 88.3%**
- **Precision (Failure): 79.8%**
- **F1 Score: 83.8%**
- **ROC-AUC: 0.94**

Compared to baseline models, this solution significantly reduced missed failures while maintaining an acceptable rate of false alarms, making it suitable for real-world deployment.

---

## Business Recommendations

- **Deploy the selected neural network model** within predictive maintenance workflows to proactively schedule inspections and repairs.
- **Tune decision thresholds dynamically** based on business tolerance:
  - Lower thresholds increase recall and reduce missed failures.
  - Higher thresholds reduce inspection costs at the risk of more missed failures.
- **Monitor model performance over time** to detect data drift and changing failure patterns.
- Use model outputs as **decision support**, not fully automated shutdown or replacement triggers.

---

## Caveats & Next Steps

- Incorporate **cost-sensitive training objectives** to directly encode the FN > FP cost hierarchy.
- Explore ensemble methods or probability calibration to improve decision reliability.
- Investigate feature attribution techniques to improve interpretability for operations teams.
- Retrain regularly as equipment ages and operating conditions evolve.

---

## Repository Structure

- `INN_ReneWind_Main_Project_FullCode_Notebook.ipynb`  
  End-to-end workflow including data preparation, model training, tuning, evaluation, and business interpretation.

---

## Tools & Technologies

- Python
- pandas, NumPy
- TensorFlow / Keras
- Neural networks
- Jupyter Notebook
