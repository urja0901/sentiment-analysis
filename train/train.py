import mlfoundry
from datetime import datetime
from train_model import train_model
from dataset import get_initial_data
import pickle

data_path = "/content/drive/MyDrive/sentiment 140/training.1600000.processed.noemoticon.csv"

X_train, X_test, y_train, y_test = get_initial_data(data_path)

model, metadata = train_model(X_train, y_train, X_test, y_test)


run = mlfoundry.get_client().create_run(project_name="sentiment-140", run_name=f"train-{datetime.now().strftime('%m-%d-%Y')}")
# run.log_params(model.get_params())

    
run.log_metrics({
    'accuracy_score': metadata["accuracy"][1]})


model_version = run.log_model(
    name="NN-classifier",
    model=model,
    framework="tensorflow",
    description="model trained for twitter sentiment analysis",
)
# model_artifact = run.log_artifact(local_path="vectorizer.pickle", artifact_path="my-artifacts")

print(f"Logged model: {model_version.fqn}")
