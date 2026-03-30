from datasets import load_dataset

# Download and load the dataset
dataset = load_dataset("HuggingFaceH4/no_robots")

# Access train/test
train_data = dataset["train"]
test_data = dataset["test"]

# Print one example
print(len(test_data))
