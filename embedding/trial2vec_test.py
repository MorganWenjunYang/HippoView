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
# Contains trial documents
test_data = {'x': data['x'].iloc[0:2]}
print('test_data', test_data['x']['reference'])
print('encode first w trials',model.encode(test_data['x']))


trialembs = download_embedding()
print(len(trialembs))
x=trialembs.search_topk('NCT01724996', k=5)
print('top 5', x)