import pickle

name = 'results_noise_0.pkl'
with open(name, 'rb') as f:
    data = pickle.load(f)

print(data)
print(data['Oracle'])
data['Oracle'] = [1e-3] * len(data['Oracle'])
print(data['Oracle'])


pickle.dump(data, open(name, 'wb'))
