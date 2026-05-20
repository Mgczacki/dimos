# PR #2189 screenshots

Example outputs of the memory2-agent tools, embedded in PR
https://github.com/dimensionalOS/dimos/pull/2189 to illustrate the
kind of multimodal tool returns the MCP-client fix enables.

| File | What | Source tool |
|---|---|---|
| `01_show_map.png` | Top-down lidar map with robot pose + a pinned waypoint | `show_map` |
| `02_frames_facing_cones.png` | Same map with viewing-cones overlay at a query point | `frames_facing` (top-down view) |
| `03_verify_room_partition.png` | Same map with 7 room polygons + per-room areas | `verify_room_partition` |
| `04_walkthrough.jpg` | Horizontal strip of camera frames across time | `walkthrough` |
| `05_frames_facing_redx.jpg` | Single camera frame with a red-X projecting the query point | `frames_facing` (per-frame view) |
