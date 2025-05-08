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

# print('test_data', test_data['x']['reference'])
# print('encode first w trials',model.encode(test_data))


trialembs = download_embedding()
print(len(trialembs))
print('top 5', trialembs.search_topk('NCT01724996', k=5))
print('top 5', trialembs.search_topk('NCT01559116', k=5))

print('NCT00795769', trialembs.search_topk('NCT00795769', k=5))
print('NCT01431274', trialembs.search_topk('NCT01431274', k=5))
print('NCT00137111', trialembs.search_topk('NCT00137111', k=5))
print('NCT00782509', trialembs.search_topk('NCT00782509', k=5))
print('NCT02105688', trialembs.search_topk('NCT02105688', k=5))
# NCT00795769 ['NCT00795769' 'NCT00343863' 'NCT03012672' 'NCT00098475' 'NCT00040846']
# NCT01431274 ['NCT01431274' 'NCT01431287' 'NCT01964352' 'NCT01703845' 'NCT02629965']
# NCT00137111 ['NCT00137111' 'NCT00549848' 'NCT03117751' 'NCT00002531' 'NCT00002756']
# NCT00782509 ['NCT00782509' 'NCT00782210' 'NCT00929851' 'NCT04402515' 'NCT01040130']
# NCT02105688 ['NCT02105688' 'NCT02105467' 'NCT02105662' 'NCT00895882' 'NCT01717326']


if __name__ == "__main__":
    print(test_data['x'].iloc[0])
    print(model.encode(test_data))
