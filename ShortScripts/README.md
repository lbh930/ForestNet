## simulate_occlusion.py
Input a point cloud file, simulate occulusion from aerial scans using Beer-Lambert law.

python simulate_occlusion.py -i L1W.laz -k 0.5

### Labels are preserved by default
- If your input LAS/LAZ already contains a per-point label dimension (e.g. `label` / `tree_id`), it will be kept automatically.
- If your labels are stored in a separate file, the script will auto-detect a sidecar next to the input named like:
	`<stem>_labels.npy`, `<stem>_labels.txt` (also supports `<stem>_label.*`, or `<stem>.npy/.txt`).
	You can also pass it explicitly via `--labels`.