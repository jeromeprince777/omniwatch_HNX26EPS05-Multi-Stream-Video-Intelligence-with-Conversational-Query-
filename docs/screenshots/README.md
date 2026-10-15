# OmniWatch UI & Architectural Evidence Dossier

High-resolution visual captures of all 12 core functionalities in **OmniWatch (HNX26EPS05)**.

| Screenshot | Feature | Description |
|---|---|---|
| **`01_search_yokohama_news_reader.png`** | **Yokohama Crossroad Camera Query** | Conversational search for *"Yokohama news reader"* resolving to Yokohama CCTV Tracklet #338 (94.2% match). |
| **`02_search_tours_bag_carrier.png`** | **Tours Plaza Bag Detection** | Grounded report for *"Show me a person carrying a bag"* at Tours Pedestrian Plaza (Tracklet #108, carries backpack, 89.1% match). |
| **`03_search_bus_truck_detection.png`** | **Large Vehicle Temporal Query** | Query *"When did a bus or truck appear?"* resolving to vehicle Tracklet #248 at 09:07:26 UTC. |
| **`04_search_east_corridor_person.png`** | **East Corridor Person Detection** | Query *"Find someone in east corridor"* resolving through Clarify-Once KB to Cam 2 Tracklet #388 (92.3% match). |
| **`05_evidence_keyframe_reticle_modal.png`** | **Cyber Reticle Keyframe Target Lock** | Evidence modal showing exact native bounding box reticle lock and frame metadata. |
| **`06_evidence_video_clip_modal.png`** | **5s On-Demand Video Player** | HTML5 video player playing the 5-second MP4 evidence clip with variable playback speed controls (`0.5x`–`2.0x`). |
| **`07_cross_camera_reid_timeline_modal.png`** | **Cross-Camera Re-ID Timeline** | Multi-hop trajectory modal reconstructing suspect path across Cam 1 -> Cam 2 -> Tours with transit times. |
| **`08_clarify_once_roi_canvas_modal.png`** | **Interactive Canvas ROI Drawing** | Clarify-Once spatial memory modal showing camera selection grid, suggestion pill, and custom canvas ROI box drawing. |
| **`09_tactical_camera_wall_grid.png`** | **4-Stream CCTV Monitoring Wall** | Synchronized 4-feed surveillance matrix (`Yokohama`, `Tours`, `Cam 1`, `Cam 2`) with live telemetry and query prefill. |
| **`10_clarify_once_knowledge_base.png`** | **Durable SQLite Spatial Memory** | Knowledge base CRUD interface displaying learned location referents, normalized tokens, and camera targets. |
| **`11_standing_surveillance_alerts_feed.png`** | **Standing Rules & Audit Event Feed** | Rule registration card, active standing monitors list, and real-time security audit event feed. |
| **`12_verified_vlm_pipeline_mode.png`** | **Verified Pipeline Mode Telemetry** | Deep VLM verification mode showing inspection telemetry breakdown (intent, object, color, latency). |
| **`demo_surveillance_session.webp`** | **Interactive Animated Demo** | Full animated screen recording of browser subagent operating the surveillance platform. |
