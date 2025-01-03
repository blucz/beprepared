import argparse
import sys
import os
import runpy
import pprint
import numpy as np
import json
import re
import textwrap
from beprepared.workspace import Workspace
from typing import Any
from colorama import Fore, Style

def highlight_property_key(key: str) -> str:
    """Highlight a property key with syntax coloring."""
    # Match tag(obj1,obj2,...) pattern, allowing hyphens, dots, and underscores in tags
    match = re.match(r'^([\w\d\-_\.]+)\((.*)\)$', key)
    if not match:
        return key
    
    tag, args = match.groups()
    
    # Try to parse each argument as JSON
    try:
        # Split on commas not inside quotes or brackets
        def split_args(s):
            parts = []
            current = ''
            depth = 0
            in_quotes = False
            
            for c in s:
                if c == '"' and (not current or current[-1] != '\\'):
                    in_quotes = not in_quotes
                elif not in_quotes:
                    if c in '[{(':
                        depth += 1
                    elif c in ']})':
                        depth -= 1
                    elif c == ',' and depth == 0:
                        parts.append(current.strip())
                        current = ''
                        continue
                current += c
            if current:
                parts.append(current.strip())
            return parts
        
        parsed_args = []
        for arg in split_args(args):
            try:
                # Parse and re-serialize to get consistent formatting
                json_obj = json.loads(arg)
                parsed_args.append(json.dumps(json_obj))
            except json.JSONDecodeError:
                parsed_args.append(arg)
        
        # Format with indentation for multiple items
        if len(parsed_args) > 1:
            highlighted = f"{Fore.LIGHTBLUE_EX}{tag}{Style.RESET_ALL}(\n"
            terminal_width = os.get_terminal_size().columns
            # Indent each argument and handle multi-line values
            for i, arg in enumerate(parsed_args):
                # Handle multi-line values with proper wrapping
                if len(arg) > terminal_width - 8:  # Account for indentation
                    wrapped = textwrap.fill(arg, terminal_width - 8)
                    indented_lines = [f"    {wrapped.splitlines()[0]}"]
                    indented_lines.extend(f"    {line}" for line in wrapped.splitlines()[1:])
                    indented_arg = '\n'.join(indented_lines)
                else:
                    indented_arg = f"    {arg}"
                highlighted += f"{Fore.GREEN}{indented_arg}{Style.RESET_ALL}"
                if i < len(parsed_args) - 1:
                    highlighted += ","
                highlighted += "\n"
            highlighted += ")"
        else:
            # Single item stays on one line
            highlighted = f"{Fore.LIGHTBLUE_EX}{tag}{Style.RESET_ALL}("
            highlighted += f"{Fore.GREEN}{parsed_args[0]}{Style.RESET_ALL}"
            highlighted += ")"
        return highlighted
    except Exception:
        return key

def format_property_value(value: Any) -> str:
    """Format a property value for display, handling different types appropriately."""
    if value is None:
        return "None"
    
    if isinstance(value, (bool, int, float, str)):
        return str(value)
        
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return f"[{len(value)} items]"
        
    if isinstance(value, dict):
        if not value:
            return "{}"
        return f"{{dict with {len(value)} items}}"
        
    if isinstance(value, np.ndarray):
        shape_str = "×".join(str(x) for x in value.shape)
        return f"ndarray({shape_str} {value.dtype})"
        
    # For other types, just show the type name and string representation
    type_name = type(value).__name__
    str_value = str(value)
    if len(str_value) > 100:
        str_value = str_value[:97] + "..."
    return f"{type_name}({str_value})"

def setup_workspace_globals(script_path=None):
    if script_path and not os.path.exists(script_path):
        print(f"Error: Script file '{script_path}' not found")
        sys.exit(1)

    # Import everything from beprepared and beprepared.nodes into globals
    import beprepared
    import beprepared.nodes
    globals_dict = {}
    
    # Import from main beprepared package
    for name in dir(beprepared):
        if not name.startswith('_'):  # Skip private/internal names
            globals_dict[name] = getattr(beprepared, name)
    
    # Import from beprepared.nodes
    for name in dir(beprepared.nodes):
        if not name.startswith('_'):  # Skip private/internal names
            globals_dict[name] = getattr(beprepared.nodes, name)
            
    return globals_dict

def run_command(args):
    # Create workspace before running the script
    with Workspace(dir=args.workspace, port=args.port) as workspace:
        globals_dict = setup_workspace_globals(args.script)
        
        # Add workspace and __file__ to globals
        globals_dict.update({
            'workspace': workspace,
            '__file__': args.script
        })
        
        # Run the script with the workspace in its globals
        runpy.run_path(args.script, init_globals=globals_dict)
        
        # Run the workspace after script evaluation
        workspace.run()

def exec_command(args):
    # Create workspace before running the code
    with Workspace(dir=args.workspace, port=args.port) as workspace:
        globals_dict = setup_workspace_globals()
        
        # Add workspace to globals
        globals_dict['workspace'] = workspace
        
        # Execute the code with the workspace in its globals
        exec(args.code, globals_dict)
        
        # Run the workspace after code execution
        workspace.run()

def main():
    parser = argparse.ArgumentParser(
        description='beprepared - AI image processing toolkit.\n\nSee https://github.com/blucz/beprepared for more info'
    )
    
    parser.add_argument('-w', '--workspace', help='Workspace directory (default: current directory)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a beprepared script')
    run_parser.add_argument('script', help='Python script to run')
    run_parser.add_argument('-p', '--port', type=int, default=8989, help='Web interface port (default: 8989)')
    
    # Exec command
    exec_parser = subparsers.add_parser('exec', help='Execute Python code')
    exec_parser.add_argument('code', help='Python code to execute')
    exec_parser.add_argument('-p', '--port', type=int, default=8989, help='Web interface port (default: 8989)')
    
    # Database commands
    db_parser = subparsers.add_parser('db', help='Database management commands')
    db_subparsers = db_parser.add_subparsers(dest='db_command', help='Database commands')
    
    
    # db list command
    list_parser = db_subparsers.add_parser('list', help='List cached properties')
    list_parser.add_argument('pattern', nargs='?', help='Only show properties matching this pattern')
    list_parser.add_argument('-d', '--domain', help='Only show properties for this domain')
    
    # db clear command
    clear_parser = db_subparsers.add_parser('clear', help='Clear cached properties')
    clear_parser.add_argument('pattern', nargs='?', help='Delete properties matching this pattern')
    clear_parser.add_argument('-d', '--domain', help='Only clear properties for this domain')
    clear_parser.add_argument('-f', '--force', action='store_true', help='Force deletion without confirmation')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.command == 'run':
        run_command(args)
    elif args.command == 'exec':
        exec_command(args)
    elif args.command == 'db':
        if not args.db_command:
            db_parser.print_help()
            sys.exit(1)
            
        workspace_dir = args.workspace or os.getcwd()
        
        if args.db_command == 'list':
            with Workspace(dir=workspace_dir) as workspace:
                props = workspace.list_props(prefix=args.pattern or '*', domain=args.domain)
                if not props:
                    print("No matching properties found.")
                else:
                    from colorama import init, Fore, Style
                    init()  # Initialize colorama
                    
                    terminal_width = os.get_terminal_size().columns
                    separator = Style.BRIGHT + ("-" * terminal_width) + Style.RESET_ALL
                    
                    # Print initial separator
                    print(separator)
                    
                    first = True
                    for key, domain, value in props:
                        if not first:
                            print(separator)
                        first = False
                        formatted_value = format_property_value(value)
                        if '\n' in formatted_value:
                            if domain:
                                print(f"{Fore.RED + Style.DIM}[{domain}]{Style.RESET_ALL} {highlight_property_key(key)}")
                            else:
                                print(f"{highlight_property_key(key)}")
                            print(f"{Style.BRIGHT}  ==>{Style.RESET_ALL}")
                            print(formatted_value)
                        else:
                            if domain:
                                print(f"{Fore.RED + Style.DIM}[{domain}]{Style.RESET_ALL} {highlight_property_key(key)}{Style.BRIGHT} -> {Style.RESET_ALL}{formatted_value}")
                            else:
                                print(f"{highlight_property_key(key)}{Style.BRIGHT} -> {Style.RESET_ALL}{formatted_value}")
                        print(f"{Style.RESET_ALL}", end='')
                    
                    # Print final separator
                    print(separator)
                        
        elif args.db_command == 'clear':
            with Workspace(dir=workspace_dir) as workspace:
                pattern = args.pattern or '*'
                domain_str = f" in domain '{args.domain}'" if args.domain else ""
                
                # Count properties before deletion
                count = workspace.count_props(pattern, args.domain)
                
                if count == 0:
                    print(f"No properties found matching pattern '{pattern}'{domain_str}.")
                    sys.exit(0)
                
                print(f"\nFound {count} properties matching pattern '{pattern}'{domain_str}.")
                
                if not args.force:
                    print(f"\n⚠️  WARNING: This will delete all {count} properties! ⚠️\n")
                    confirm = input("Type 'yes' (without quotes) to confirm deletion: ")
                    if confirm.lower() != 'yes':
                        print("Aborted.")
                        sys.exit(1)
                        
                deleted = workspace.clear_props(pattern, args.domain)
                print(f"Deleted {deleted} properties matching pattern '{pattern}'{domain_str}.")

if __name__ == '__main__':
    main()
