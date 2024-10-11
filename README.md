# LogicBasedAI
- "Logic-Based Explainable and Incremental Machine Learning"
- Gupta, Gopal and Wang, Huaduo and Basu, Kinjal and Shakerin, Farahad and Padalkar, Parth and Salazar, Elmer and Varanasi, Sarat Chandra and Dasgupta, Sopam
- https://ojs.aaai.org/index.php/AAAI-SS/article/view/27678

###  Abstract:
Mainstream machine learning methods lack interpretability, explainability, incrementality, and data-economy. We propose using logic programming (LP) to rectify these problems. We discuss the FOLD family of rule-based machine learning algorithms that learn models from relational datasets as a set of default rules. These models are competitive with state-of-the-art machine learning systems in terms of accuracy and execution efficiency. We also motivate how logic programming can be useful for theory revision and explanation-based learning.

## Instructions

### Set up - Install Depenencies
```bash
pip install -r requirements.txt
```

### Data Processing
The ```DataProcessing``` class is used to load, preprocess, and split the data. 

Example:
```python
from dataPorcessing import DataProcessing

# Initialize DataProcessing
data_processor = DataProcessing("Path_to_your_csv_file.csv")

# Load and Preprocess data sets
data_processor.load_data()
data_processor.preprocess()

# Split data into training and test sets.
X_train, X_test, y_train, y_test = data_processor.split_data(target_col) # target_col is the dependent variable (label we are solving for)
```

### Model Definition
The ```RuleBasedModel``` class is used to define, solve, and retrieve results from a rule based logic programming machine learning model. 

Example:
```python
from modelDefinition import RuleBasedModel

# Initialize the model
model = RuleBasedModel(name="My Model", sense=pulp.LpMaximize)

# Define variables
var_dict = {
    "x1": (0, None, "Continuous"),
    "x2": (0, None, "Continuous")
}
model.define_vars(var_dict)

# Set objective function
obj_coeffs = {
    "x1": 1,
    "x2": 2
}
model.set_objective(obj_coeffs)

# Add constraints
constraints = [
    ({"x1": 1, "x2": 1}, pulp.LpConstraintLE, 10),
    ({"x1": 1}, pulp.LpConstraintGE, 0),
    ({"x2": 1}, pulp.LpConstraintGE, 0)
]
model.add_constraints(constraints)

# Solve the model
model.solve()

# Get results
results = model.get_results()
print("Results:", results)
```

### Test Coverage
- Run tests in test folder with: ```coverage run -m unittest discover -s test```
- Get coverage report with : ```coverage report -m ```
