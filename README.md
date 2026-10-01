# noface

Anonymize every face in a video except the people you want to keep visible.

You give it a video and a photo, or a folder of photos. Every face found in those photos stays visible. Everyone else is blurred, covered with a black rectangle, or pixelated. The original audio is copied onto the result. Detection, recognition, and rendering all run on your machine.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

[ffmpeg](https://ffmpeg.org/) is optional and worth having. With it, phone videos are rotated upright and the original audio is kept. Without it, the video is written with no audio.

The first run downloads the YuNet face detector and the SFace recognizer from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (about 40 MB) into `~/.noface/models`.

## Use

```bash
noface --video birthday.mp4 --references subject.jpg --output birthday-anon.mp4

noface --video street.mov --references ./people_to_keep --output street-anon.mp4 --style black

noface --video clip.mp4 --references me.png --output clip-pixel.mp4 --style pixelate
```

`--style` is `blur` (the default), `black`, or `pixelate`.

Check a reference folder before a long video:

```bash
noface --references ./people_to_keep --dry-run
```

### Reference photos

- One image keeps every face in that image. A group photo keeps the whole group.
- A folder keeps every face in every image, including images in subfolders. Use one clear photo per person, or several photos of the same person. Extra angles make a turned head easier to recognize.
- Files and folders whose names start with `.` are skipped.

### How a face is judged

Each face in the video is turned into an identity vector and compared with the reference photos. The score is SFace cosine similarity.

- **0.363 or higher** (`--confirm-threshold`): this is someone to keep. The track stays visible for the next 30 frames (`--track-ttl`) even when recognition drops, for example during a turn, motion blur, or a short occlusion.
- **0.28 up to 0.363** (`--gray-threshold`): the face stays visible for this frame. A gray score does not start that memory on its own.
- **Below 0.28**: the face is anonymized, unless the track was confirmed recently.

Lower `--gray-threshold` when a kept person is sometimes anonymized. Raise it when a bystander is sometimes left visible.

`--pad` (default `0.35`) grows each anonymized box so the forehead and chin are covered. `--blur-strength` (default `16`) controls blur: `8` is mild and `32` is heavy. `--pixel-size` is the block size for pixelation.

### Detector

YuNet looks at the frame, and at a smaller copy of the frame, so a face that fills the picture is still found. `--max-det-side` (default `1920`) caps the larger pass. On 4K footage, faces smaller than about 40 pixels can be missed; raise `--max-det-side` when those matter. `--min-size` (default `12`) ignores tiny boxes, and `--det-score` (default `0.6`) is the detector confidence.

The speed is printed at the end. Every frame is processed on the CPU, so a long video takes a while.

## Develop

```bash
pip install -e ".[dev]"
pytest
```

`NOFACE_SMOKE=1 pytest tests/test_smoke.py` runs YuNet and SFace on a real frame. It downloads two OpenCV sample photos on first use.

## Demo

`demo/` is a nine-second excerpt from the official theatrical trailer of *Zindagi Na Milegi Dobara* (Zoya Akhtar, released 15 July 2011). Excel Movies posted the trailer: [Zindagi Na Milegi Dobara- Official Theatrical Trailer](https://www.youtube.com/watch?v=ifIBOKCfjVs). The clip is 1:55 to 2:04, with the letterbox cropped off. The film is still under copyright.

Abhay Deol, Hrithik Roshan, and Katrina Kaif stay visible. Farhan Akhtar is covered, along with everyone else. The stills in `demo/references/` are from other moments of the same trailer. `demo/side-by-side.mp4` plays the original and the result side by side.

```bash
noface --video demo/source.mp4 \
  --references demo/references \
  --output demo/output.mp4 \
  --style black \
  --confirm-threshold 0.42 \
  --gray-threshold 0.38 \
  --track-ttl 12 \
  --max-gap 12
```
