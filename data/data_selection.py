import sys
from pathlib import Path

input_path = Path("sys.argv[1]")
output_path = input_path.with_stem(input_path.stem + "_liver_only")

lines = input_path.read_text().splitlines()
filtered = [line for line in lines if "liver" in line.lower()]

output_path.write_text("\n".join(filtered))
print(f"Kept {len(filtered)}/{len(lines)} lines → {output_path}")