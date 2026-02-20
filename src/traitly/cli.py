# traitly/cli.py

"""
Command Line Interface for Traitly
Allows analyzing fruit images directly from terminal.

Usage:
  traitly --fruit_internal -i PATH -o PATH_OUTPUT --json PATH/JSON --num_cores 1
  traitly --fruit_external -i PATH -o PATH_OUTPUT --json PATH/JSON --num_cores 1
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from tabulate import tabulate as _tabulate
    def _fmt_examples(rows):
        return _tabulate(rows, tablefmt='plain')
except ImportError:
    def _fmt_examples(rows):
        return "\n".join(f"  {r[0]}" for r in rows)


# ============================================================================
# Parser
# ============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        prog='traitly',
        description='Traitly - Fruit Phenotyping Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
            "For more info: https://github.com/mariameraz/traitly",
            "",
        ])
    )

    # ── Mode (mutually exclusive) ────────────────────────────────────────────
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

    # ── Required ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        metavar='PATH',
        help='Path to image file or folder'
    )

    # ── Optional ─────────────────────────────────────────────────────────────
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
    path = Path(path_str)
    if not path.exists():
        print(f">> Error: Path does not exist: {path_str}")
        sys.exit(1)
    return path


def _validate_json(json_str: str) -> str:
    if json_str is not None and not Path(json_str).exists():
        print(f">> Error: JSON config file does not exist: {json_str}")
        sys.exit(1)
    return json_str


# ============================================================================
# Internal analysis
# ============================================================================

def run_internal(args):
    """Run FruitInternalAnalyzer.analyze_folder() or process a single file."""
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

def run_external(args):
    """Run FruitExternalAnalyzer.analyze_folder() or process a single file."""
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

def main():
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