"""`motionly` command-line interface."""

from pathlib import Path

import typer

from motionly_engine import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"motionly {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """Motionly engine: measured timing and easing from motion graphics videos."""


@app.command()
def ease(spec: str = typer.Argument(..., help="Ease name (CSS/Penner/GSAP) or x1,y1,x2,y2.")) -> None:
    """Show an ease's parameters and its nearest named preset."""
    from motionly_engine.fit import easing

    if "," in spec:
        try:
            params = tuple(float(v) for v in spec.split(","))
            easing.bezier(0.5, params)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        typer.echo(f"cubic-bezier{params}")
        curve = params
    else:
        name = easing.GSAP_ALIASES.get(spec, spec)
        if name not in easing.EXACT and name not in easing.PRESETS:
            raise typer.BadParameter(f"unknown ease {spec!r}")
        kind = "exact" if name in easing.EXACT else "bezier"
        approx = easing.PRESETS.get(name)
        typer.echo(f"{name} ({kind})" + (f", bezier approx {approx}" if approx else ""))
        curve = name
    m = easing.nearest_preset(curve)
    typer.echo(f"nearest: {m.name} ({m.kind}), dist {m.dist:.3f}")


@app.command()
def generate(
    count: int = typer.Option(40, help="Number of clips."),
    seed: int = typer.Option(1, help="Random seed."),
    out: Path = typer.Option(Path("data/gt"), help="Output directory."),
    size: str = typer.Option("mixed", help="960x540, 1920x1080, WxH, or mixed (alternates 960x540 / 1920x1080)."),
    fps: int = typer.Option(30, help="Frame rate: 24, 30 or 60."),
    crf: int = typer.Option(26, help="H.264 CRF."),
) -> None:
    """Render ground-truth clips (clip_NNN.mp4 + clip_NNN.json)."""
    from motionly_engine.generate import FPS_CHOICES, generate_clips

    if fps not in FPS_CHOICES:
        raise typer.BadParameter(f"--fps must be one of {FPS_CHOICES}")

    def progress(scene) -> None:
        n_tw = sum(len(e.tweens) for e in scene.elements)
        tag = " stagger" if scene.groups else ""
        typer.echo(f"{scene.id}  {scene.width}x{scene.height}  {scene.duration_s:.2f}s  "
                   f"{len(scene.elements)} el  {n_tw} tw{tag}")

    stats = generate_clips(count, seed, out, size=size, fps=fps, crf=crf, on_clip=progress)
    mix = ", ".join(f"{k} {v}" for k, v in sorted(stats["eases"].items()))
    typer.echo(f"{stats['clips']} clips, {stats['tweens']} tweens, {stats['stagger']} stagger, "
               f"eases: {mix}, {stats['seconds']:.1f}s total -> {out}")


@app.command()
def measure(
    clip: Path = typer.Argument(..., exists=True, help="Video file."),
    out: Path = typer.Option(None, help="Output dir (default data/runs/<clip-stem>)."),
    plot: bool = typer.Option(False, help="Plot truth vs measured (needs clip_NNN.json beside the clip)."),
    element: int = typer.Option(0, help="Truth element index to plot."),
) -> None:
    """Measure per-element x, y, scale, opacity signals from a clip."""
    import json
    import time

    from motionly_engine.measure import measure as run_measure

    t0 = time.time()
    sig = run_measure(clip)
    out = out or Path("data/runs") / clip.stem
    sig.save(out / "signals.json")
    typer.echo(f"{clip.name}: {len(sig.timestamps)} frames, {len(sig.elements)} elements, "
               f"{time.time() - t0:.1f}s -> {out / 'signals.json'}")
    if plot:
        from motionly_engine.generate.scene import Scene
        from motionly_engine.measure.plot import plot_element, truth_vs_measured

        truth = clip.with_suffix(".json")
        if not truth.exists():
            raise typer.BadParameter(f"--plot needs {truth}")
        t, series, title, _ = truth_vs_measured(sig, Scene.from_dict(json.loads(truth.read_text())), element)
        png = plot_element(t, series, title, out / f"plot_el{element}.png")
        typer.echo(f"plot -> {png}")
