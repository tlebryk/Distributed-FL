# %%
from benchmark import HumanEvalBenchmark
from client import FederatedClient
import logging

# logging.basicConfig(level=logging.DEBUG)

human_eval = HumanEvalBenchmark()
client = FederatedClient(client_id="client_1", server_address="localhost:50051")
human_eval.load_dataset()
client.initialize_model()
client.initialize_tokenizer()
model = client.model
tokenizer = client.tokenizer
# get first three rows of dataset
human_eval.dataset = human_eval.dataset.select(range(5))
# %%
results = human_eval.run(client.model, client.tokenizer)
results
