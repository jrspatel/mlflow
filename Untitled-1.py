# %%
"""
    Initializing MLflow
    logging params
    storing and loading the model
"""

import mlflow 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

# %%
mlflow.autolog()

# %%
db = load_diabetes()

# %%
# splitting the data 
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# %%
print("dataset shape:" , db.data.shape)
print("training data shape:" , X_train.shape)
print("testing data shape:" , X_test.shape)

# %%
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3) 
rf.fit(X_train,y_train)

# %%
# predictions 
predictions = rf.predict(X_test)

# %% [markdown]
# #### Signatures are used to store model inputs, outputs and model parameters

# %%
# storing the model 

from mlflow.models import infer_signature 

signature = infer_signature(X_test,predictions)
mlflow.sklearn.log_model(rf,'model4',signature=signature)

# %%
with mlflow.start_run():
    mlflow.sklearn.log_model(rf, "iris_rf1", signature=signature)

# %%
mlflow.end_run()

# %%
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# %%
print(mlflow.__version__)

# %%
import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run() as run:
    # Load the diabetes dataset.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    print(predictions)

    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)

    print(f"Run ID: {run.info.run_id}")

# %% [markdown]
# #### loading the model 

# %%


import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("mlruns/0/9eae74ce8fe64387a7a31a560c0b759d/artifacts/model4")
predictions = model.predict(X_test)
print(predictions)

# %%
run = mlflow.active_run()
print(run)

# %%
run = mlflow.active_run()
print(f"run_id: {run.info.run_id}; status: {run.info.status}")


# %%

mlflow.end_run()

# %%
run = mlflow.get_run(run.info.run_id)
print(f"run_id: {run.info.run_id}; status: {run.info.status}")
print("--")

# Check for any active runs
print(f"Active run: {mlflow.active_run()}")

# %%



