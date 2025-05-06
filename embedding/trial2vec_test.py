# some weird cuda issue

import torch
import numpy
# Fix serialization issues with NumPy arrays
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
torch.serialization.safe_globals([numpy.ndarray])
from trial2vec import Trial2Vec, load_demo_data, download_embedding



# Always use CPU
device = torch.device('cpu')

# Initialize model on CPU
model = Trial2Vec(device="cpu")
model.from_pretrained()

data = load_demo_data()
print(data)

# Contains trial documents
test_data = {'x': data['x']} 

# Make prediction
pred = model.predict(test_data)
emb = model.encode(data['x'])



# from trial2vec import download_embedding
# import torch
# import numpy
# torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
# torch.serialization.safe_globals([numpy.ndarray])
# trialembs = download_embedding()
# print(len(trialembs))
# trialembs.search_topk('NCT01724996', k=5)