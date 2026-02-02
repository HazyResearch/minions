#!/usr/bin/env python3
"""
Graphical configuration launcher for FinanceBench Evaluator.

This script provides a Tkinter-based GUI for configuring the evaluator,
similar to Linux kernel's xconfig/guiconfig.

Usage:
    python config/kguiconfig.py                     # Use default paths
    python config/kguiconfig.py -c config/.config   # Load existing config
"""

import os
import sys
import argparse
from pathlib import Path

# TclError for handling Tkinter errors
try:
    from tkinter import TclError
except ImportError:
    # If tkinter isn't available, create a dummy exception class
    class TclError(Exception):
        pass


def main():
    """Launch guiconfig for evaluator configuration."""
    script_dir = Path(__file__).parent.absolute()
    default_config = str(script_dir / '.config')
    
    parser = argparse.ArgumentParser(
        description="GUI configuration for FinanceBench Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python config/kguiconfig.py                       # Configure from scratch
    python config/kguiconfig.py -c config/.config     # Load existing config
    python config/kguiconfig.py --defconfig defconfig # Start from preset
        """
    )
    parser.add_argument(
        '-c', '--config',
        default=default_config,
        help=f'Path to .config file (default: {default_config})'
    )
    parser.add_argument(
        '--defconfig',
        help='Load a defconfig preset before launching (e.g., defconfig)'
    )
    
    args = parser.parse_args()
    
    # Determine Kconfig file location
    kconfig_path = script_dir / 'Kconfig'
    
    if not kconfig_path.exists():
        print(f"Error: Kconfig file not found at {kconfig_path}")
        sys.exit(1)
    
    try:
        import kconfiglib
    except ImportError:
        print("Error: kconfiglib is not installed.")
        print("Please install it with: pip install kconfiglib")
        sys.exit(1)
    
    # Set environment variables for Kconfig
    os.environ['srctree'] = str(script_dir)
    os.environ['KCONFIG_CONFIG'] = args.config
    
    # Load Kconfig
    kconf = kconfiglib.Kconfig(str(kconfig_path))
    
    # Load defconfig if specified
    if args.defconfig:
        defconfig_path = script_dir / 'presets' / f'{args.defconfig}_defconfig'
        if defconfig_path.exists():
            print(f"Loading defconfig: {defconfig_path}")
            kconf.load_config(str(defconfig_path))
        else:
            # Try without _defconfig suffix
            defconfig_path = script_dir / 'presets' / args.defconfig
            if defconfig_path.exists():
                print(f"Loading defconfig: {defconfig_path}")
                kconf.load_config(str(defconfig_path))
            else:
                print(f"Warning: Defconfig '{args.defconfig}' not found in {script_dir / 'presets'}")
    
    # Load existing config if it exists
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading existing config: {config_path}")
        kconf.load_config(str(config_path))
    
    # Try to run guiconfig
    try:
        # guiconfig imports tkinter at module level, so ImportError happens here
        import guiconfig as guiconfig_module
        # The function is called menuconfig() in guiconfig module
        guiconfig_module.menuconfig(kconf)
    except ImportError as e:
        if "tkinter" in str(e).lower():
            print(f"Error: Tkinter is not installed: {e}")
            print("\nThe kconfiglib GUI requires Tkinter. Install it with:")
            print("  - Ubuntu/Debian: sudo apt install python3-tk")
            print("  - Fedora: sudo dnf install python3-tkinter")
            print("  - macOS: brew install python-tk")
        else:
            print(f"Error: guiconfig module not available: {e}")
            print("\nMake sure kconfiglib is installed: pip install kconfiglib")
        print("\nAlternatively, use kmenuconfig for terminal-based configuration:")
        print(f"    python config/kmenuconfig.py -c {args.config}")
        sys.exit(1)
    except TclError as e:
        print(f"Error: Cannot initialize GUI (no display?): {e}")
        print("\nUse kmenuconfig for terminal-based configuration:")
        print(f"    python config/kmenuconfig.py -c {args.config}")
        sys.exit(1)
    except Exception as e:
        if "display" in str(e).lower() or "DISPLAY" in str(e):
            print(f"Error: No display available for GUI: {e}")
            print("\nUse kmenuconfig for terminal-based configuration:")
            print(f"    python config/kmenuconfig.py -c {args.config}")
            sys.exit(1)
        raise
    
    # Save configuration
    kconf.write_config(args.config)
    print(f"\nConfiguration saved to: {args.config}")
    print(f"\nTo run evaluation:")
    print(f"    make run")
    print(f"  or:")
    print(f"    python evaluate/financebench_evaluator.py {args.config}")


if __name__ == '__main__':
    main()
