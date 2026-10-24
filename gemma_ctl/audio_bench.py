"""Spoken-command benchmark for the voice path: Piper TTS clips (clean + 10 dB noise) ->
Gemma 4 E4B transcription (word error rate) -> 26B agent (dry run) -> right tool + move?

  python -m gemma_ctl.audio_bench        (needs runs/gemma/audio/.venv_tts + voice)
"""
import json, re, subprocess, time
from pathlib import Path
from .chat_web import transcribe
from .agent import Agent, OLLAMA
from .real_robot import RealExecutor, real_system, real_tool_schemas

D = Path(__file__).resolve().parents[1] / "runs/gemma/audio"
CASES = [  # (spoken text, expected tool, expected name or None)
    ("Hey robot, can you wave at me?", "gesture", "wave"),
    ("Blow me a kiss.", "gesture", "blow_kiss"),
    ("Make a heart with your hands.", "gesture", "heart"),
    ("Give me a high five!", "gesture", "high_five"),
    ("Do the macarena.", "dance", "macarena"),
    ("Show me some gangnam style.", "dance", "gangnam_style"),
    ("Let's see you moonwalk.", "move", "moonwalk"),
    ("Can you dance the salsa?", "move", "salsa"),
    ("Walk like you hurt your leg.", "move", "injured leg walk"),
    ("Walk on crutches.", "move", "crutch walk"),
    ("Do a catwalk for me.", "move", "catwalk"),
    ("Go for a jog.", "move", "jog"),
    ("Crouch down and walk.", "move", "crouch walk"),
    ("Do some squats.", "move", "squat"),
    ("Show me some shadow boxing.", "move", "shadow boxing"),
    ("Throw a roundhouse kick.", "move", "roundhouse kick"),
    ("Can you do a high jump?", "play_motion", None),
    ("What do you see?", "look", None),
    ("What moves do you know?", None, None),
    ("Stop!", "stop", None),
]


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def tts(text, out):
    subprocess.run([str(D / ".venv_tts/bin/piper"), "-m", str(D / "voices/en_US-lessac-medium.onnx"),
                    "-f", str(out)], input=text.encode(), check=True, capture_output=True)


def noisy(src, out, snr_db=10):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "lavfi", "-i",
                    "anoisesrc=color=pink:amplitude=1", "-filter_complex",
                    f"[1]volume=-{snr_db + 18}dB[n];[0][n]amix=inputs=2:duration=first:normalize=0",
                    "-ar", "16000", "-ac", "1", str(out)], check=True)


def route(agent, text):
    n0 = len(agent.trace)
    agent.run(text)
    calls = [t for t in agent.trace[n0:] if "tool" in t and t["tool"] != "find_motions"]
    return (calls[0]["tool"], calls[0]["args"].get("name")) if calls else (None, None)


def main():
    ex = RealExecutor("sim", live=False)
    rows = []
    for i, (text, tool, name) in enumerate(CASES):
        clean = D / f"cmd{i:02d}.wav"; noise = D / f"cmd{i:02d}_noise10.wav"
        if not clean.exists():
            tts(text, clean)
        if not noise.exists():
            noisy(clean, noise)
        for cond, wav in (("clean", clean), ("noise10dB", noise)):
            heard, s = transcribe(wav.read_bytes())
            agent = Agent("gemma4:26b", ex, {}, OLLAMA, log=lambda *a: None,
                          tools=real_tool_schemas("sim"), system=real_system("sim"))
            got = route(agent, heard) if heard else (None, None)
            ok_route = got[0] == tool and (name is None or got[1] == name)
            rows.append(dict(text=text, cond=cond, heard=heard, wer=round(wer(text, heard), 2),
                             hear_s=s, tool=got[0], name=got[1], route_ok=ok_route))
            print(f"{cond:9s} wer {rows[-1]['wer']:.2f} {s:4.1f}s {'OK ' if ok_route else 'BAD'} "
                  f"heard={heard!r:45.45} -> {got}", flush=True)
    (D / "bench.json").write_text(json.dumps(rows, indent=1))
    for cond in ("clean", "noise10dB"):
        r = [x for x in rows if x["cond"] == cond]
        print(f"{cond}: mean WER {sum(x['wer'] for x in r)/len(r):.3f}, routed right {sum(x['route_ok'] for x in r)}/{len(r)}, "
              f"hear {sum(x['hear_s'] for x in r)/len(r):.2f} s")


if __name__ == "__main__":
    main()
