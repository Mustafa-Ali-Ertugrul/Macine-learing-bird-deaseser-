import os

base = 'final_dataset_split'
results = []
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(base, split)
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    results.append(f'\n{split.upper()}:')
    total = 0
    for cls in classes:
        count = len(os.listdir(os.path.join(split_dir, cls)))
        total += count
        results.append(f'  {cls}: {count}')
    results.append(f'  TOTAL: {total}')

with open('image_counts.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print('Done - see image_counts.txt')
