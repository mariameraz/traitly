# traitly/cli.py

"""
Command Line Interface for Traitly
Allows analyzing fruit images directly from terminal.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog='traitly',
        description='Traitly - Fruit Internal Structure Phenotyping Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  traitly internal-structure tests/sample_data/DP14-106.jpg
  
  # Analyze folder with 4 cores
  traitly internal-structure tests/sample_data/ --cores 4
  
  # Analyze with custom output directory
  traitly internal-structure tests/sample_data/ -o results/
  
  # Skip label detection
  traitly internal-structure image.jpg --no-label
  
  # Use custom reference diameter
  traitly internal-structure image.jpg --diameter 3.0

For more info: https://github.com/mariameraz/traitly
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Analysis type')
    
    # ========================================================================
    # SUBCOMMAND: internal-structure (internal structure analysis)
    # ========================================================================
    internal_parser = subparsers.add_parser(
        'internal-structure',
        help='Analyze internal fruit structure (locules, pericarp, symmetry)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  traitly internal-structure image.jpg
  traitly internal-structure folder/ --cores 4 --output results/
        """
    )
    
    # Required argument
    internal_parser.add_argument(
        'path',
        type=str,
        help='Path to image file or folder'
    )
    
    # Output options
    internal_parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: same as input for single image, "Results/" for folder)'
    )
    
    # Processing options
    internal_parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='Number of CPU cores for parallel processing (default: 1)'
    )
    
    internal_parser.add_argument(
        '--stamp',
        action='store_true',
        help='Invert image colors (for images with black background)'
    )
    
    internal_parser.add_argument(
        '--no-label',
        action='store_true',
        help='Skip QR code and label detection'
    )
    
    # Calibration options
    setup_group = internal_parser.add_argument_group('setup options')
    
    setup_group.add_argument(
        '--diameter',
        type=float,
        default=2.5,
        help='Reference circle diameter in cm (default: 2.5)'
    )
    
    setup_group.add_argument(
        '--width',
        type=float,
        default=None,
        help='Image width in cm for manual calibration'
    )
    
    setup_group.add_argument(
        '--height',
        type=float,
        default=None,
        help='Image height in cm for manual calibration'
    )
    
    # Segmentation options
    seg_group = internal_parser.add_argument_group('segmentation options')
    
    seg_group.add_argument(
        '--min-circularity',
        type=float,
        default=0.3,
        help='Minimum circularity for fruit detection (0-1, default: 0.3)'
    )
    
    seg_group.add_argument(
        '--min-locule-area',
        type=int,
        default=300,
        help='Minimum locule area in pixels (default: 300)'
    )
    
    setup_group.add_argument(
    '--gpu',
    action='store_true',
    help='Use GPU for label detection (requires CUDA-compatible GPU)'
    )

    seg_group.add_argument(
        '--merge-locules',
        action='store_true',
        help='Merge close locules automatically'
    )
    
    seg_group.add_argument(
        '--kernel-size',
        type=int,
        default=5,
        help='Morphological kernel size (default: 5)'
    )
    
    # Analysis options
    analysis_group = internal_parser.add_argument_group('analysis options')
    
    analysis_group.add_argument(
        '--contour-mode',
        type=str,
        choices=['raw', 'hull', 'approx', 'ellipse', 'circle'],
        default='raw',
        help='Contour transformation mode (default: raw)'
    )
    
    analysis_group.add_argument(
        '--symmetry-shifts',
        type=int,
        default=100,
        help='Number of shifts for angular symmetry (default: 100)'
    )
    
    analysis_group.add_argument(
        '--pericarp-rays',
        type=int,
        default=360,
        help='Number of rays for pericarp thickness (default: 360)'
    )
    
    # Visualization options
    viz_group = internal_parser.add_argument_group('visualization options')
    
    viz_group.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not display annotated images'
    )
    
    viz_group.add_argument(
        '--font-scale',
        type=float,
        default=1.5,
        help='Font scale for annotations (default: 1.5)'
    )
    
    viz_group.add_argument(
        '--label-position',
        type=str,
        choices=['top', 'bottom', 'left', 'right'],
        default='top',
        help='Position of fruit labels (default: top)'
    )
    
    # Verbose/quiet options
    internal_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    internal_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    return parser


def analyze_internal(args):
    """Execute internal structure analysis."""
    from .internal_structure import FruitAnalyzer
    
    # Validate path
    path = Path(args.path)
    if not path.exists():
        print(f">> Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    try:
        analyzer = FruitAnalyzer(str(path))
    except Exception as e:
        print(f">> Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Determine if single image or folder
    is_folder = path.is_dir()
    
    if is_folder:
        # ====================================================================
        # FOLDER ANALYSIS
        # ====================================================================
        if not args.quiet:
            print(f"- Analyzing folder with {args.cores} core(s)...")
        
        try:
            analyzer.analyze_folder(
                output_dir=args.output,
                stamp=args.stamp,
                contour_mode=args.contour_mode,
                n_kernel=args.kernel_size,
                min_circularity=args.min_circularity,
                min_locule_area=args.min_locule_area,
                merge_locules=args.merge_locules,
                n_shifts=args.symmetry_shifts,
                num_rays=args.pericarp_rays,
                n_cores=args.cores,
                diameter_cm=args.diameter,
                width_cm=args.width,
                length_cm=args.height,
                detect_label=not args.no_label,
                font_scale=args.font_scale,
                label_position=args.label_position
            )
            
        except Exception as e:
            print(f">> Error during folder analysis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    else:
        # ====================================================================
        # SINGLE IMAGE ANALYSIS
        # ====================================================================
        if not args.quiet:
            print("- Analyzing single image...")
        
        try:
            analyzer.read_image()
            
            analyzer.setup_measurements(
                diameter_cm=args.diameter,
                width_cm=args.width,
                length_cm=args.height,
                label=not args.no_label,
                verbose=args.verbose,
                plot=False, gpu=args.gpu
            )
            
            analyzer.create_mask(
                stamp=args.stamp,
                n_kernel=args.kernel_size,
                plot=False
            )
            
            analyzer.find_fruits(
                min_circularity=args.min_circularity,
                output_message=args.verbose
            )
            
            results = analyzer.analyze_image(
                plot=not args.no_plot,
                contour_mode=args.contour_mode,
                stamp=args.stamp,
                min_locule_area=args.min_locule_area,
                merge_locules=args.merge_locules,
                n_shifts=args.symmetry_shifts,
                num_rays=args.pericarp_rays,
                font_scale=args.font_scale,
                label_position=args.label_position,
            )
            
            output_dir = args.output if args.output else str(path.parent / "output")
            results.save_all(output_dir=output_dir, output_message=not args.quiet)
            
            if not args.quiet:
                print("- Single image analysis completed!")
                print(f"Results saved to: {output_dir}")
                
        except Exception as e:
            print(f">> Error during image analysis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == 'internal-structure':
        analyze_internal(args)
    else:
        print(f">> Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()