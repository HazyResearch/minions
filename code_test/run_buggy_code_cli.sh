#!/bin/bash

# Set default values
PYTHON="python3"
INPUT_FILE=""
OUTPUT_DIR="output"
THRESHOLD=0.7
MAX_ITERATIONS=3
VERBOSE=""

# Display usage information
function show_usage {
    echo "Usage: $0 [options] <input_file>"
    echo "Options:"
    echo "  -o, --output DIR       Output directory (default: 'output')"
    echo "  -t, --threshold NUM    Pass rate threshold 0.0-1.0 (default: 0.7)"
    echo "  -m, --max-iter NUM     Maximum fix iterations (default: 3)"
    echo "  -k, --api-key KEY      OpenAI API key (optional, can use env var)"
    echo "  -v, --verbose          Enable verbose output"
    echo "  -h, --help             Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -m|--max-iter)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        -k|--api-key)
            export OPENAI_API_KEY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
                shift
            else
                echo "Too many arguments: $1"
                show_usage
            fi
            ;;
    esac
done

# Check if input file is provided
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: Input file is required"
    show_usage
fi

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Warning: OPENAI_API_KEY environment variable is not set"
    echo "You can set it with --api-key or by using:"
    echo "export OPENAI_API_KEY=your_api_key"
    read -p "Continue without API key? (y/n): " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        exit 1
    fi
fi

# Run the CLI
CMD="$PYTHON buggy_code_cli.py $INPUT_FILE --output-dir $OUTPUT_DIR --threshold $THRESHOLD --max-iterations $MAX_ITERATIONS $VERBOSE"
echo "Running: $CMD"
eval $CMD 