"""`motionly` command-line interface."""

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
