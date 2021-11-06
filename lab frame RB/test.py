import pickle

with open('1000.pkl', 'rb') as f:
    x = pickle.load(f)
f.close()

print(x)
print(x[0])
print(x[1])
