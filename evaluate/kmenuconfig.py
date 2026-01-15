#!/usr/bin/env python3
"""
Interactive menuconfig launcher for FinanceBench Evaluator.

This script provides a terminal-based (ncurses) interface for configuring
the evaluator, similar to Linux kernel's menuconfig.

Usage:
    python evaluate/kmenuconfig.py              # Use default paths
    python evaluate/kmenuconfig.py -c .config   # Load existing config
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    """Launch menuconfig for evaluator configuration."""
    parser = argparse.ArgumentParser(
        description="Interactive configuration for FinanceBench Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate/kmenuconfig.py                    # Configure from scratch
    python evaluate/kmenuconfig.py -c my.config       # Load existing config
    python evaluate/kmenuconfig.py --defconfig cvoc   # Start from preset
        """
    )
    parser.add_argument(
        '-c', '--config',
        default='.config',
        help='Path to .config file (default: .config in current directory)'
    )
    parser.add_argument(
        '--defconfig',
        help='Load a defconfig preset before launching (e.g., cvoc_minimal)'
    )
    
    args = parser.parse_args()
    
    # Determine Kconfig file location
    script_dir = Path(__file__).parent.absolute()
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
        defconfig_path = script_dir / 'configs' / f'{args.defconfig}_defconfig'
        if defconfig_path.exists():
            print(f"Loading defconfig: {defconfig_path}")
            kconf.load_config(str(defconfig_path))
        else:
            # Try without _defconfig suffix
            defconfig_path = script_dir / 'configs' / args.defconfig
            if defconfig_path.exists():
                print(f"Loading defconfig: {defconfig_path}")
                kconf.load_config(str(defconfig_path))
            else:
                print(f"Warning: Defconfig '{args.defconfig}' not found in {script_dir / 'configs'}")
    
    # Load existing config if it exists
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading existing config: {config_path}")
        kconf.load_config(str(config_path))
    
    # Try to run menuconfig
    try:
        import menuconfig as menuconfig_module
        menuconfig_module.menuconfig(kconf)
    except ImportError:
        # Fall back to text-based configuration
        print("\nMenuconfig TUI not available. Running text-based configuration...")
        _text_config(kconf)
    except Exception as e:
        if "terminal" in str(e).lower() or "curses" in str(e).lower():
            print(f"\nTerminal does not support ncurses: {e}")
            print("Running text-based configuration instead...")
            _text_config(kconf)
        else:
            raise
    
    # Save configuration
    kconf.write_config(args.config)
    print(f"\nConfiguration saved to: {args.config}")
    print(f"\nTo run evaluation:")
    print(f"    python evaluate/financebench_evaluator.py {args.config}")


def _text_config(kconf):
    """Simple text-based configuration fallback."""
    print("\n" + "=" * 60)
    print("Text-Based Configuration")
    print("=" * 60)
    print("\nType 'y' for yes, 'n' for no, or press Enter for default.")
    print("Type a value for string/int options, or press Enter for default.")
    print("Type 'q' to finish and save.\n")
    
    def configure_menu(menu, indent=0):
        """Recursively configure menu items."""
        prefix = "  " * indent
        
        for node in menu.list:
            if node.item == kconfiglib.MENU:
                print(f"\n{prefix}=== {node.prompt[0]} ===")
                configure_menu(node, indent + 1)
            elif node.item == kconfiglib.COMMENT:
                print(f"{prefix}# {node.prompt[0]}")
            elif isinstance(node.item, kconfiglib.Symbol):
                sym = node.item
                if sym.visibility == 0:
                    continue
                    
                prompt = node.prompt[0] if node.prompt else sym.name
                
                if sym.type == kconfiglib.BOOL:
                    default = 'y' if sym.tri_value == 2 else 'n'
                    val = input(f"{prefix}[{default}] {prompt} (y/n/q): ").strip().lower()
                    if val == 'q':
                        return False
                    if val in ('y', 'yes'):
                        sym.set_value(2)
                    elif val in ('n', 'no'):
                        sym.set_value(0)
                
                elif sym.type == kconfiglib.TRISTATE:
                    default = {0: 'n', 1: 'm', 2: 'y'}.get(sym.tri_value, 'n')
                    val = input(f"{prefix}[{default}] {prompt} (y/m/n/q): ").strip().lower()
                    if val == 'q':
                        return False
                    if val in ('y', 'yes'):
                        sym.set_value(2)
                    elif val == 'm':
                        sym.set_value(1)
                    elif val in ('n', 'no'):
                        sym.set_value(0)
                
                elif sym.type in (kconfiglib.STRING, kconfiglib.INT, kconfiglib.HEX):
                    default = sym.str_value
                    val = input(f"{prefix}[{default}] {prompt}: ").strip()
                    if val == 'q':
                        return False
                    if val:
                        sym.set_value(val)
        
        return True
    
    import kconfiglib
    configure_menu(kconf.top_node)


if __name__ == '__main__':
    main()

