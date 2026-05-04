# traitly/cli.py

"""
Command-line interface for Traitly fruit phenotyping.

Provides the ``traitly`` entry point for running internal and external
fruit analysis pipelines directly from the terminal, on single images
or entire folders.

Usage
-----
.. code-block:: bash

    traitly --fruit_internal -i PATH [-o PATH] [--json PATH] [--num_cores N]
    traitly --fruit_external -i PATH [-o PATH] [--json PATH] [--num_cores N]

For folder inputs, delegates to
:meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.analyze_folder`
or
:meth:`~traitly.fruit_phenotyping.external_analysis.FruitExternalAnalyzer.analyze_folder`.
For single images, delegates to
:meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.process_single_file`.
"""

import argparse
import sys
import os
from pathlib import Path
from rich_argparse import RawDescriptionRichHelpFormatter

# ============================================================================
# Parser
# ============================================================================

def _fmt_examples(rows):
    return "\n".join(f"  {r[0]}" for r in rows)

def create_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser for Traitly.

    Defines a mutually exclusive mode group (``--fruit_internal`` /
    ``--fruit_external``), required and optional arguments, and a
    formatted epilog with usage examples.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready to call ``.parse_args()`` on.
    """
    parser = argparse.ArgumentParser(
        prog='traitly',
        description=(
            "              . ݁₊ ⊹ . ݁ ⟡ ݁ Traitly ⟡ ݁. ⊹ ₊ .  ݁\n"
            " Computer vision toolkit for high-throughput fruit phenotyping"
        ),
        formatter_class=RawDescriptionRichHelpFormatter,
        epilog="\n".join([
            "",
            "Examples:",
            _fmt_examples([
                ("  # Internal structure analysis (single image or folder)",),
                ("  traitly --fruit_internal -i tests/sample_data/",),
                ("  traitly --fruit_internal -i tests/sample_data/ -o results/ --num_cores 4",),
                ("  traitly --fruit_internal -i tests/sample_data/ --json config.json",),
                ("",),
                ("  # External analysis (single image or folder)",),
                ("  traitly --fruit_external -i tests/sample_data/",),
                ("  traitly --fruit_external -i tests/sample_data/ -o results/ --json config.json --num_cores 4",),
            ]),
            "",
            "="*70,
            "For more details ->",
            "   - GitHub Repository: https://github.com/mariameraz/traitly \n"
            "   - Documentation: https://traitly.readthedocs.io/ ",
            "="*70,
            "",
        ])
    )

    # Mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--fruit_internal',
        action='store_true',
        help='Analyze internal fruit structure (locules, pericarp, symmetry)'
    )
    mode_group.add_argument(
        '--fruit_external',
        action='store_true',
        help='Analyze external fruit structure (morphology, color)'
    )

    # Required
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        metavar='PATH',
        help='Path to image file or folder'
    )

    # Optional
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        metavar='PATH',
        help='Output directory (default: <input>/Results/)'
    )

    parser.add_argument(
        '--json',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to JSON config file with analysis parameters'
    )

    parser.add_argument(
        '--num_cores',
        type=int,
        default=1,
        metavar='N',
        help='Number of CPU cores for parallel processing (default: 1)'
    )

    parser.add_argument(
        '--no_morphology',
        action='store_true',
        help='Skip morphology analysis'
    )

    parser.add_argument(
        '--no_color',
        action='store_true',
        help='Skip color analysis'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    return parser


# ============================================================================
# Helpers
# ============================================================================

def _validate_input(path_str: str) -> Path:
    """
    Validate that the input path exists and return it as a :class:`Path`.

    Prints an error message and calls ``sys.exit(1)`` if the path does
    not exist.

    Parameters
    ----------
    path_str : str
        Raw input path string from the CLI argument.

    Returns
    -------
    Path
        Validated :class:`~pathlib.Path` object.
    """
    path = Path(path_str)
    if not path.exists():
        print(f">> Error: Path does not exist: {path_str}")
        sys.exit(1)
    return path


def _validate_json(json_str: str) -> str:
    """
    Validate that the JSON config file exists if provided.

    Prints an error message and calls ``sys.exit(1)`` if ``json_str``
    is not ``None`` and the file does not exist.

    Parameters
    ----------
    json_str : str or None
        Path to the JSON config file, or ``None`` if not provided.

    Returns
    -------
    str or None
        The original ``json_str`` if valid, or ``None``.
    """
    if json_str is not None and not Path(json_str).exists():
        print(f">> Error: JSON config file does not exist: {json_str}")
        sys.exit(1)
    return json_str


# ============================================================================
# Internal analysis
# ============================================================================

def run_internal(args: argparse.Namespace) -> None:
    """
    Run the internal fruit analysis pipeline from the CLI.

    Instantiates :class:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer`
    and dispatches to:

    - :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.analyze_folder`
      when ``args.input`` is a directory.
    - :meth:`~traitly.fruit_phenotyping.internal_analysis.FruitInternalAnalyzer.process_single_file`
      when ``args.input`` is a single image file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing ``input``, ``output``, ``json``,
        ``num_cores``, ``no_morphology``, and ``no_color``.
    """
    from traitly.fruit_phenotyping.internal_analysis import FruitInternalAnalyzer

    path     = _validate_input(args.input)
    json_path = _validate_json(args.json)

    try:
        analyzer = FruitInternalAnalyzer(str(path))
    except Exception as e:
        print(f">> Error initializing FruitInternalAnalyzer: {e}")
        sys.exit(1)

    analyze_morphology = not args.no_morphology
    analyze_color      = not args.no_color

    if path.is_dir():
        try:
            analyzer.analyze_folder(
                analyze_morphology=analyze_morphology,
                analyze_color=analyze_color,
                json_path=json_path,
                output_path=args.output,
                num_cores=args.num_cores,
                verbose=True,
            )
        except Exception as e:
            print(f">> Error during internal folder analysis: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    else:
        # Single image — process_single_file
        import json as _json
        config = {}
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = _json.load(f) or {}

        try:
            analyzer.load_image(plot=False)
            df_m, df_c, err, n_fruits, ann_img = analyzer.process_single_file(
                config=config,
                json_path=None,
                analyze_morphology=analyze_morphology,
                analyze_color=analyze_color,
                save_image=True,
                output_path=args.output,
            )
            if err:
                print(f">> Error processing image: {err.get('status', 'Unknown error')}")
                sys.exit(1)
            print(f"Done. Fruits detected: {n_fruits}")
        except Exception as e:
            print(f">> Error during internal image analysis: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)


# ============================================================================
# External analysis
# ============================================================================

def run_external(args: argparse.Namespace) -> None:
    """
    Run the external fruit analysis pipeline from the CLI.

    Instantiates :class:`~traitly.fruit_phenotyping.external_analysis.FruitExternalAnalyzer`
    and dispatches to:

    - :meth:`~traitly.fruit_phenotyping.external_analysis.FruitExternalAnalyzer.analyze_folder`
      when ``args.input`` is a directory.
    - :meth:`~traitly.fruit_phenotyping.external_analysis.FruitExternalAnalyzer.process_single_file`
      when ``args.input`` is a single image file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing ``input``, ``output``, ``json``,
        ``num_cores``, ``no_morphology``, and ``no_color``.
    """
    from traitly.fruit_phenotyping.external_analysis import FruitExternalAnalyzer

    path      = _validate_input(args.input)
    json_path = _validate_json(args.json)

    try:
        analyzer = FruitExternalAnalyzer(str(path))
    except Exception as e:
        print(f">> Error initializing FruitExternalAnalyzer: {e}")
        sys.exit(1)

    analyze_morphology = not args.no_morphology
    analyze_color      = not args.no_color

    if path.is_dir():
        try:
            analyzer.analyze_folder(
                analyze_morphology=analyze_morphology,
                analyze_color=analyze_color,
                json_path=json_path,
                output_path=args.output,
                num_cores=args.num_cores,
                verbose=True,
            )
        except Exception as e:
            print(f">> Error during external folder analysis: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    else:
        # Single image
        import json as _json
        config = {}
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = _json.load(f) or {}

        try:
            analyzer.load_image(plot=False)
            df_m, df_c, err, n_fruits, ann_img = analyzer.process_single_file(
                config=config,
                json_path=None,
                analyze_morphology=analyze_morphology,
                analyze_color=analyze_color,
                save_image=True,
                output_path=args.output,
            )
            if err:
                print(f">> Error processing image: {err.get('status', 'Unknown error')}")
                sys.exit(1)
            print(f"Done. Fruits detected: {n_fruits}")
        except Exception as e:
            print(f">> Error during external image analysis: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """
    Entry point for the ``traitly`` CLI command.

    Parses arguments via :func:`create_parser` and dispatches to
    :func:`run_internal` or :func:`run_external` based on the selected
    mode flag. Prints help and exits cleanly if neither flag is set.
    """
    parser = create_parser()
    args   = parser.parse_args()

    if args.fruit_internal:
        run_internal(args)
    elif args.fruit_external:
        run_external(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
