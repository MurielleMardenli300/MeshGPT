import pickle

path = "liver_dataset.pkl"

with open(path, "rb") as f:
    obj = pickle.load(f)

print("Type:", type(obj))

if hasattr(obj, "__len__"):
    print("Length:", len(obj))

if isinstance(obj, dict):
    print("First keys:", list(obj.keys())[:10])
    print('\n---')
    for i in range(len(obj['vertices_train'])):
        print(obj[f'name_train'][i].split('_')[0])
        
    print('\n---\n')
    vertices = [obj[f'vertices_train'][i] for i in range(len(obj[f'vertices_train']))]
    print(len(vertices))