"""
NeuroQuant CLI - main command line interface.

Commands:
- encode: Encode video with NeuroQuant pipeline
- benchmark: Comparative testing of methods
- report: Generate reports from results
- analyze: Analyze video complexity
"""

import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel

from .encoder import Codec
from .utils import (
    load_config,
    get_video_info,
    parse_bitrate,
    format_bitrate,
    check_ffmpeg,
    check_cuda,
    log_info,
    log_success,
    log_error,
    log_warning,
    ensure_dir,
)

console = Console()


def print_banner():
    """Print program banner."""
    banner = r"""
[bold cyan]============================================================
     _   _                      ___                    _
    | \ | | ___ _   _ _ __ ___ / _ \ _   _  __ _ _ __ | |_
    |  \| |/ _ \ | | | '__/ _ \ | | | | | |/ _` | '_ \| __|
    | |\  |  __/ |_| | | | (_) | |_| | |_| | (_| | | | | |_
    |_| \_|\___|\__,_|_|  \___/ \__\_\\__,_|\__,_|_| |_|\__|

    [bold yellow]Intelligent Video Compression System[/bold yellow]
============================================================[/bold cyan]
"""
    console.print(banner)


@click.group()
@click.version_option(version="0.1.0", prog_name="NeuroQuant")
def cli():
    """NeuroQuant - Intelligent Video Compression System.

    Combines per-frame rate control based on R-lambda theory
    with Real-ESRGAN post-processing.
    """
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--bitrate", "-b", default="1M", help="Target bitrate (e.g. 1M, 500k)")
@click.option("--sr/--no-sr", default=False, help="Apply Real-ESRGAN SR")
@click.option("--sr-threshold", default=70.0, help="VMAF threshold for SR activation")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def encode(
    input_path: str,
    output_path: str,
    bitrate: str,
    sr: bool,
    sr_threshold: float,
    config: Optional[str],
    quiet: bool,
):
    """Encode video using NeuroQuant pipeline.

    Example:
        neuroquant encode input.mp4 output.mp4 --bitrate 1M --sr
    """
    if not quiet:
        print_banner()

    # Check ffmpeg
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    if not ffmpeg_ok:
        log_error(ffmpeg_msg)
        sys.exit(1)

    log_info(f"FFmpeg: {ffmpeg_msg}")

    # Load config
    cfg = load_config(config)

    # Parse bitrate
    target_bitrate = parse_bitrate(bitrate)

    # Get video info
    video_info = get_video_info(input_path)
    log_info(f"Input video: {Path(input_path).name}")
    log_info(f"Resolution: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.0f}fps")
    log_info(f"Duration: {video_info['duration']:.1f} sec ({video_info['frame_count']} frames)")
    log_info(f"Target bitrate: {format_bitrate(target_bitrate)}")

    # Step 1: Complexity analysis
    log_info("\n[Step 1/4] Analyzing frame complexity...")
    from .analyzer import ComplexityAnalyzer
    analyzer = ComplexityAnalyzer(
        spatial_weight=cfg.get("complexity", {}).get("spatial_weight", 0.4),
        temporal_weight=cfg.get("complexity", {}).get("temporal_weight", 0.5),
        cut_weight=cfg.get("complexity", {}).get("cut_weight", 0.1),
        scene_threshold=cfg.get("complexity", {}).get("scene_threshold", 27.0),
        analysis_scale=cfg.get("complexity", {}).get("analysis_scale", 0.5),
    )
    complexity_data = analyzer.analyze(input_path, show_progress=not quiet)

    # Step 2: Generate QP plan
    log_info("\n[Step 2/4] Generating QP plan...")
    from .controller import RLambdaController
    controller = RLambdaController(
        alpha=cfg.get("rate_control", {}).get("alpha", 6.7542),
        beta=cfg.get("rate_control", {}).get("beta", -1.7860),
        qp_min=cfg.get("rate_control", {}).get("qp_min", 10),
        qp_max=cfg.get("rate_control", {}).get("qp_max", 51),
        delta_max=cfg.get("rate_control", {}).get("delta_max", 8),
        i_frame_bonus=cfg.get("rate_control", {}).get("i_frame_bonus", 4),
        gop_seconds=cfg.get("rate_control", {}).get("gop_seconds", 2.0),
    )
    qp_plan = controller.generate_qp_plan(
        complexity_data,
        target_bitrate=target_bitrate,
        fps=video_info["fps"],
        width=video_info["width"],
        height=video_info["height"],
    )

    # Step 3: Encoding
    log_info("\n[Step 3/4] Encoding video...")
    from .encoder import FFmpegEncoder
    encoder = FFmpegEncoder(
        codec=Codec.HEVC,
        preset=cfg.get("encoder", {}).get("preset", "medium"),
        disable_aq=cfg.get("encoder", {}).get("disable_aq", True),
    )

    if sr:
        # Encode to temp file
        temp_output = str(Path(output_path).with_suffix(".temp.mp4"))
        encode_result = encoder.encode_with_qp_plan(
            input_path, temp_output, qp_plan,
            target_bitrate=target_bitrate, show_progress=not quiet
        )
    else:
        encode_result = encoder.encode_with_qp_plan(
            input_path, output_path, qp_plan,
            target_bitrate=target_bitrate, show_progress=not quiet
        )

    if not encode_result.success:
        log_error(f"Encoding error: {encode_result.error_message}")
        sys.exit(1)

    # Step 4: SR post-processing (if needed)
    if sr:
        log_info("\n[Step 4/4] SR post-processing...")

        cuda_ok, cuda_msg = check_cuda()
        if cuda_ok:
            log_info(f"GPU: {cuda_msg}")
        else:
            log_warning("CUDA not available, SR will run on CPU")

        from .sr_processor import SRPostProcessor
        sr_processor = SRPostProcessor(
            vmaf_threshold=sr_threshold,
            model_name=cfg.get("sr", {}).get("model", "RealESRNet_x2plus"),
            tile_size=cfg.get("sr", {}).get("tile_size", 512),
        )

        sr_result = sr_processor.process_video(
            temp_output, input_path, output_path, show_progress=not quiet
        )

        # Remove temp file
        Path(temp_output).unlink(missing_ok=True)

        if not sr_result.success:
            log_error(f"SR error: {sr_result.error_message}")
            sys.exit(1)

        log_info(f"SR processed frames: {sr_result.frames_processed}/{sr_result.frames_total}")
    else:
        log_info("\n[Step 4/4] SR skipped")

    # Final info
    output_info = get_video_info(output_path)
    output_size = Path(output_path).stat().st_size / (1024 * 1024)

    log_success(f"\nEncoding complete!")
    log_info(f"Output file: {output_path}")
    log_info(f"Size: {output_size:.2f} MB")
    log_info(f"Actual bitrate: {format_bitrate(output_info.get('bitrate', 0))}")


@cli.command()
@click.argument("video_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default="./results", help="Output directory")
@click.option("--methods", "-m", default="h264,hevc,nq", help="Methods (comma-separated)")
@click.option("--bitrates", "-b", default="300k,600k,1200k", help="Bitrates (comma-separated)")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
def benchmark(
    video_dir: str,
    output: str,
    methods: str,
    bitrates: str,
    config: Optional[str],
):
    """Run comparative benchmark of encoding methods.

    Example:
        neuroquant benchmark ./test_videos -m h264,hevc,nq -b 300k,600k,1200k
    """
    print_banner()

    # Check ffmpeg
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    if not ffmpeg_ok:
        log_error(ffmpeg_msg)
        sys.exit(1)

    # Parse parameters
    method_list = [m.strip() for m in methods.split(",")]
    bitrate_list = [parse_bitrate(b.strip()) for b in bitrates.split(",")]

    # Find video files
    video_dir = Path(video_dir)
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    video_files = [
        str(f) for f in video_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ]

    if not video_files:
        log_error(f"No video files found in {video_dir}")
        sys.exit(1)

    log_info(f"Found videos: {len(video_files)}")
    log_info(f"Methods: {', '.join(m.upper() for m in method_list)}")
    log_info(f"Bitrates: {', '.join(format_bitrate(b) for b in bitrate_list)}")

    # Run benchmark
    from .benchmark import BenchmarkEngine
    engine = BenchmarkEngine(
        methods=method_list,
        bitrates=bitrate_list,
        config_path=config,
    )

    report = engine.run(video_files, output, show_progress=True)

    log_success(f"\nBenchmark complete!")
    log_info(f"Results: {output}/benchmark_report.json")


@cli.command()
@click.argument("json_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="./report", help="Output directory")
@click.option("--format", "-f", "fmt", default="html,png", help="Formats (html,png,pdf)")
def report(json_path: str, output: str, fmt: str):
    """Generate report from benchmark results.

    Example:
        neuroquant report results/benchmark_report.json -o ./report -f html,png
    """
    print_banner()

    formats = [f.strip() for f in fmt.split(",")]

    from .report import ReportGenerator
    generator = ReportGenerator()
    created_files = generator.generate_from_json(json_path, output, formats)

    log_success(f"\nReport generated!")
    for f in created_files:
        log_info(f"  - {f}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="JSON file to save results")
@click.option("--plot", is_flag=True, help="Create complexity plot")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
def analyze(
    input_path: str,
    output: Optional[str],
    plot: bool,
    config: Optional[str],
):
    """Analyze video complexity without encoding.

    Example:
        neuroquant analyze input.mp4 --output complexity.json --plot
    """
    print_banner()

    cfg = load_config(config)

    video_info = get_video_info(input_path)
    log_info(f"Video: {Path(input_path).name}")
    log_info(f"Resolution: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.0f}fps")

    from .analyzer import ComplexityAnalyzer
    analyzer = ComplexityAnalyzer(
        spatial_weight=cfg.get("complexity", {}).get("spatial_weight", 0.4),
        temporal_weight=cfg.get("complexity", {}).get("temporal_weight", 0.5),
        cut_weight=cfg.get("complexity", {}).get("cut_weight", 0.1),
        scene_threshold=cfg.get("complexity", {}).get("scene_threshold", 27.0),
        analysis_scale=cfg.get("complexity", {}).get("analysis_scale", 0.5),
    )

    results = analyzer.analyze(input_path, show_progress=True)

    # Save JSON
    if output:
        analyzer.save_to_json(results, output)

    # Create plot
    if plot:
        import matplotlib.pyplot as plt
        import numpy as np

        complexities = [r.complexity for r in results]
        frames = np.arange(len(complexities))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(frames, complexities, alpha=0.3)
        ax.plot(frames, complexities, linewidth=0.5)

        # Mark scene cuts
        cuts = [r.frame_idx for r in results if r.is_scene_cut]
        for cut in cuts:
            ax.axvline(cut, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Complexity")
        ax.set_title(f"Frame Complexity: {Path(input_path).name}")
        ax.set_xlim(0, len(complexities))
        ax.set_ylim(0, 1)

        plot_path = output.replace(".json", "_plot.png") if output else "complexity_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        log_info(f"Plot saved: {plot_path}")

    log_success("Analysis complete!")


@cli.command()
def info():
    """Show system and dependencies information."""
    print_banner()

    console.print(Panel.fit("[bold]System Check[/bold]", style="cyan"))

    # FFmpeg
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    status = "[green][OK][/green]" if ffmpeg_ok else "[red][X][/red]"
    console.print(f"  {status} FFmpeg: {ffmpeg_msg}")

    # CUDA
    cuda_ok, cuda_msg = check_cuda()
    status = "[green][OK][/green]" if cuda_ok else "[yellow][-][/yellow]"
    console.print(f"  {status} CUDA: {cuda_msg}")

    # Python packages
    console.print("\n[bold]Installed packages:[/bold]")

    packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("scenedetect", "scenedetect"),
        ("rich", "rich"),
        ("click", "click"),
    ]

    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            console.print(f"  [green][OK][/green] {name}: {version}")
        except ImportError:
            console.print(f"  [red][X][/red] {name}: not installed")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
