import os

dataset_path = '/projects/bgbh/datasets/VoxCeleb/wav' 

# get counts for every folder
all_counts = []
for spk_id in os.listdir(dataset_path):
    spk_path = os.path.join(dataset_path, spk_id)
    if os.path.isdir(spk_path):
        # Count .wav files
        count = sum([len(files) for r, d, files in os.walk(spk_path)])
        all_counts.append((spk_id, count))

# sort by count: largest folders first
all_counts.sort(key=lambda x: x[1], reverse=True)

# pick the winners until we hit 10,000
total_files = 0
min_celebrities = []

for spk_id, count in all_counts:
    if total_files < 10000:
        min_celebrities.append((spk_id, count))
        total_files += count
    else:
        break

print(f"To get {total_files} files, we only need {len(min_celebrities)} celebrities.")
print("Top 5 contributors in VoxCeleb1:")
for i in range(min(5, len(min_celebrities))):
    print(f"  {min_celebrities[i][0]}: {min_celebrities[i][1]} files")

# save this to a file
with open('minimal_10k_list.txt', 'w') as f:
    for spk_id, count in min_celebrities:
        f.write(f"{spk_id}\n")