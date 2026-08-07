# Customer-Purchase-Prediction-using-KNN
# Customer Purchase Predictor (KNN)

A Python implementation of the **K-Nearest Neighbors (KNN)** classification algorithm. This project predicts customer purchasing decisions based on **Age** and **Estimated Salary**, featuring a custom inference function and an automated **Meshgrid decision boundary visualization**.

## 📌 Project Overview
* **Target Variable**: Customer purchase behavior (Binary classification: Buy / No Buy).
* **Predictor Features**: Two distinct numerical scales: Age (years) and Estimated Salary (\$).
* **Core Toolkit**: Built using `Numpy` for grid arrays, `Scikit-Learn` for model fitting, and `Matplotlib` for boundary plots.

## 📂 Repository Structure
```text
├── main.py              # Complete executable script containing training and inference
├── requirements.txt     # Project dependency specifications
└── README.md            # Project documentation and guide
```

## 🛠️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com
   cd knn-purchase-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. **Execute the script**
   ```bash
   python main.py
   ```

## 🧠 Technical Implementation

### The Meshgrid Boundary Map
To expose the exact logic of the KNN classifier, the script evaluates a dense coordinate matrix over the feature space:
* **Grid Generation**: Utilizes `np.meshgrid` to generate dense pixel coordinates matching feature ranges.
* **Continuous Prediction**: Flattens the grid coordinates using `np.c_` to run batch classification.
* **Visual Isolation**: Reshapes predictions back into the grid dimensions to map clear "Buy" vs "No Buy" zones via `plt.contourf`.

### Random Input Inference Function
The custom prediction function handles vector formatting on the fly, allowing immediate testing with random real-world variables:

```python
def predict_purchase(age, salary):
    """Formats raw inputs and queries the trained KNN model."""
    user_input = np.array([[age, salary]])
    prediction = knn.predict(user_input)
    
    if prediction == 1:
        return "🔮 Prediction: This customer WILL buy the product!"
    return "🔮 Prediction: This customer WILL NOT buy the product."

# Test execution
print(predict_purchase(40, 85000))
```

## 💡 Key Machine Learning Insights
* **Feature Scale Impact**: KNN relies purely on Euclidean distance metrics.
* **The Variance Issue**: Unscaled Salary numbers heavily dominate Age scales during distance calculation.
* **Next Steps**: Implementing `StandardScaler` is required to eliminate mathematical bias towards higher-magnitude features.

## 🤝 Contributing
* **Issues**: Open an issue to report bugs or suggest enhancements.
* **Pull Requests**: Submissions for scaling improvements or custom distance metrics are welcome.
