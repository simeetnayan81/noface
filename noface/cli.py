"""Command-line interface for noface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from noface import __version__
from noface.engine import OpenCVFaceEngine, default_model_dir
from noface.errors import NofaceError
from noface.gallery import enroll, log_enrollment, reference_images
from noface.pipeline import anonymize_video
from noface.types import Options


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def build_parser() -> argparse.ArgumentParser:
    defaults = Options()
    parser = argparse.ArgumentParser(
        prog="noface",
        description=(
            "Anonymize faces in a video and leave chosen people visible. "
            "Every face found in a reference image, or in the images inside a reference folder, is kept. "
            "Everyone else is blurred, covered with a black box, or pixelated."
        ),
        epilog=(
            "examples:\n"
            "  noface --video birthday.mp4 --references subject.jpg --output birthday-anon.mp4\n"
            "  noface --video street.mov --references ./people_to_keep --output street-anon.mp4 --style black\n"
            "  noface --video clip.mp4 --references me.png friends/ --output clip-pixel.mp4 --style pixelate\n"
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"noface {__version__}")
    source = parser.add_argument_group("input and output")
    source.add_argument("--video", type=Path, help="Video to anonymize.")
    source.add_argument(
        "--references",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Photo, or folder of photos, of the people to keep. "
            "Every detected face in those images stays visible. "
            "Several photos of the same person make the match more reliable."
        ),
    )
    source.add_argument("--output", "-o", type=Path, help="Where to write the anonymized video.")
    source.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect faces in the reference images, then exit.",
    )
    source.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. 0 processes the whole video.",
    )
    source.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop the original audio instead of copying it to the output.",
    )

    look = parser.add_argument_group("anonymization")
    look.add_argument(
        "--style",
        choices=("blur", "black", "pixelate"),
        default=defaults.style,
        help="How to hide faces that are not in the references.",
    )
    look.add_argument(
        "--blur-strength",
        type=int,
        default=defaults.blur_strength,
        help="How hard to blur. 8 is mild, 16 hides a face, 32 is very strong.",
    )
    look.add_argument(
        "--pixel-size",
        type=int,
        default=defaults.pixel_size,
        help="Pixel block size, in pixels, for --style pixelate.",
    )
    look.add_argument(
        "--pad",
        type=float,
        default=defaults.pad,
        help="Expand each anonymized box by this fraction of its size on every side.",
    )

    matching = parser.add_argument_group("matching")
    matching.add_argument(
        "--confirm-threshold",
        type=float,
        default=defaults.confirm_threshold,
        help=(
            "Cosine similarity that confirms a face and remembers that track. "
            "0.363 is the SFace same-person threshold."
        ),
    )
    matching.add_argument(
        "--gray-threshold",
        type=float,
        default=defaults.gray_threshold,
        help=(
            "Faces at least this similar stay visible, without remembering the track. "
            "Lower values keep more borderline faces visible."
        ),
    )
    matching.add_argument(
        "--track-ttl",
        type=int,
        default=defaults.track_ttl,
        help=(
            "After a confirmed match, keep that tracked face visible for this many "
            "later frames where recognition drops."
        ),
    )
    matching.add_argument(
        "--max-gap",
        type=int,
        default=defaults.max_gap,
        help="Frames a face can disappear and still count as the same track.",
    )
    matching.add_argument(
        "--hold-frames",
        type=int,
        default=defaults.hold_frames,
        help="Keep drawing an anonymized box for this many frames after a stranger is missed.",
    )

    detector = parser.add_argument_group("detector")
    detector.add_argument(
        "--det-score",
        type=float,
        default=defaults.det_score,
        help="Minimum detector confidence. Lower values find more faces and more false alarms.",
    )
    detector.add_argument(
        "--min-size",
        type=float,
        default=defaults.min_size,
        help="Ignore video faces smaller than this many pixels.",
    )
    detector.add_argument(
        "--max-det-side",
        type=int,
        default=defaults.max_det_side,
        help="Longest side, in pixels, of the larger detection pass.",
    )
    detector.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir(),
        help="Where YuNet and SFace weights are stored. They are downloaded on first use.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log detection scales and debug detail.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    try:
        return _run(args)
    except NofaceError as exc:
        logging.getLogger("noface").error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.getLogger("noface").error("Stopped before the output was written.")
        return 130


def cli() -> None:
    sys.exit(main())


def _run(args: argparse.Namespace) -> int:
    _validate(args)
    if args.dry_run:
        engine = _engine(args)
        enrollment = enroll(reference_images(list(args.references)), engine, det_score=args.det_score)
        log_enrollment(enrollment)
        return 0
    if args.video is None or args.output is None:
        raise NofaceError("Pass --video and --output, or use --dry-run to only check the reference photos.")
    if not args.video.is_file():
        raise NofaceError(f"Video not found: {args.video}")
    if args.video.resolve() == args.output.resolve():
        raise NofaceError("Choose an output path that is different from the input video.")
    engine = _engine(args)
    anonymize_video(
        args.video,
        list(args.references),
        args.output,
        engine,
        _options(args),
        max_frames=args.max_frames or None,
        keep_audio=not args.no_audio,
    )
    return 0


def _validate(args: argparse.Namespace) -> None:
    if args.gray_threshold > args.confirm_threshold:
        raise NofaceError("--gray-threshold cannot be higher than --confirm-threshold.")
    if not -1.0 <= args.gray_threshold <= 1.0 or not -1.0 <= args.confirm_threshold <= 1.0:
        raise NofaceError("Thresholds must be between -1 and 1.")
    if args.track_ttl < 0 or args.max_gap < 0 or args.hold_frames < 0:
        raise NofaceError("--track-ttl, --max-gap, and --hold-frames cannot be negative.")
    if args.hold_frames > args.max_gap:
        raise NofaceError("--hold-frames cannot be greater than --max-gap.")
    if args.blur_strength < 1:
        raise NofaceError("--blur-strength must be at least 1.")
    if args.pixel_size < 2:
        raise NofaceError("--pixel-size must be at least 2.")
    if args.pad < 0:
        raise NofaceError("--pad cannot be negative.")
    if args.max_frames < 0:
        raise NofaceError("--max-frames cannot be negative.")
    if not 0.0 <= args.det_score <= 1.0:
        raise NofaceError("--det-score must be between 0 and 1.")
    if args.min_size < 0:
        raise NofaceError("--min-size cannot be negative.")
    if args.max_det_side < 64:
        raise NofaceError("--max-det-side must be at least 64.")
    for path in args.references:
        if not path.exists():
            raise NofaceError(f"Reference not found: {path}")


def _options(args: argparse.Namespace) -> Options:
    return Options(
        style=args.style,
        blur_strength=args.blur_strength,
        pixel_size=args.pixel_size,
        pad=args.pad,
        confirm_threshold=args.confirm_threshold,
        gray_threshold=args.gray_threshold,
        track_ttl=args.track_ttl,
        max_gap=args.max_gap,
        hold_frames=args.hold_frames,
        det_score=args.det_score,
        min_size=args.min_size,
        max_det_side=args.max_det_side,
    )


def _engine(args: argparse.Namespace) -> OpenCVFaceEngine:
    return OpenCVFaceEngine(
        model_dir=args.model_dir,
        det_score=args.det_score,
        max_det_side=args.max_det_side,
    )
