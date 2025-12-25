import torch

chk = torch.load('Models/best_model.pth', map_location='cpu')
print('Checkpoint info:')
print(f'Type: {type(chk)}')

if isinstance(chk, dict):
    print(f'Keys: {chk.keys()}')
    print(f'Num classes: {chk.get("num_classes")}')
    print(f'Labels: {chk.get("labels")}')
    print(f'Number of labels: {len(chk.get("labels", []))}')
else:
    print('Legacy model format')
    if hasattr(chk, 'f3'):
        print(f'Output features: {chk.f3.out_features}')
