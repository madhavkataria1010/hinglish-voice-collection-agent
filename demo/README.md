# Recorded demo

Place the recorded conversation here as `recording.wav` (or `.mp4`).

## What this needs to be

- **Length:** at least 90 seconds
- **Code switches:** at least 3 between English and Hindi
- **Mid-number switch:** at least 1 (e.g. _"I can pay pachas thousand rupees"_)
- **Source:** real microphone input through the live agent (not text simulation)

## How I record it

```bash
make tunnel    # remote GPU STT
make web       # http://localhost:7860/
```

Open the browser, start the call, and screen-record system audio + mic
(QuickTime "New Screen Recording" with audio source set, or any equivalent).

## Why this isn't checked in

The recording is large and personal-voice — it's gitignored
(`demo/recording.wav` and `demo/*.mp4` are listed in `.gitignore` and
`.dockerignore`). The submission instructions say to attach the file with
the form, not push it to the repo.
