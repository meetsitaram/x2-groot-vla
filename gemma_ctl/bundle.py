"""Task-run bundle: one folder holding everything needed to regenerate videos.

  <bundle>/
    rig.log  frames/  cam_poses.jsonl     the recording (tripod camera, per-frame pose)
    trajectory.jsonl                      25 Hz robot root + 31 joints + prop poses (kinematic replay)
    live/  world_timeline.jsonl  conversation.jsonl  s*_look/ s*_refine*/ ...   Gemma's perception
    transcript.json  agent.log            Gemma's calls and replies
    run_bundle.json                       manifest (this script)

Re-render with tweaks (no sim, no Gemma):
    python gemma_ctl/overlay_video.py <bundle> <bundle>/live        # world-model boxes + HUD
    REVIEW_FRAMES=frames_ov python gemma_ctl/make_review.py <bundle>  # videos + clips

  python gemma_ctl/bundle.py <bundle> [--note "..."]
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def git(repo):
    try:
        rev = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        br = subprocess.run(["git", "-C", repo, "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip().splitlines()
        return {"repo": repo, "branch": br, "rev": rev, "modified_files": dirty}
    except Exception as e:  # noqa: BLE001
        return {"repo": repo, "error": repr(e)}


def main():
    b = Path(sys.argv[1])
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
    def n_lines(p):
        return sum(1 for _ in open(p)) if p.exists() else 0
    conv = [json.loads(l) for l in open(b / "live/conversation.jsonl")] if (b / "live/conversation.jsonl").exists() else []
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
        "requests": [e["text"] for e in conv if e["kind"] == "user"],
        "files": {"frames": len(list((b / "frames").glob("f*.jpg"))), "cam_poses": n_lines(b / "cam_poses.jsonl"),
                  "trajectory_25hz": n_lines(b / "trajectory.jsonl"),
                  "world_snapshots": n_lines(b / "live/world_timeline.jsonl"), "conversation_events": len(conv),
                  "surveys": sorted(p.name for p in (b / "live").glob("s*") if p.is_dir())},
        "model": "gemma4:26b (Ollama)",
        "code": [git("/home/stickbot/projects/x2-groot-vla"), git("/home/stickbot/Projects/GR00T-WBC-gemma")],
        "scene": "gemma_ctl/kitchen_scene_v1.json",
        "rerender": {"overlay": "python gemma_ctl/overlay_video.py <bundle> <bundle>/live",
                     "videos": "REVIEW_FRAMES=frames_ov python gemma_ctl/make_review.py <bundle>",
                     "new_camera": "kinematic replay of trajectory.jsonl (gemma_ctl/replay_render.py, Isaac)"},
    }
    (b / "run_bundle.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man["files"]), "->", b / "run_bundle.json")


if __name__ == "__main__":
    main()
