#!/bin/bash

ORIGINAL_INPUTS_DIR=benchmark
INSTRUMENTED_INPUTS_DIR="${ORIGINAL_INPUTS_DIR}_instrumented"
TESTGEN_FUZZ_OUTPUT_DIR="${ORIGINAL_INPUTS_DIR}_fuzzer_tests"
TESTGEN_GA_OUTPUT_DIR="${ORIGINAL_INPUTS_DIR}_ga_tests"
LOG_FILE_FUZZ="${TESTGEN_FUZZ_OUTPUT_DIR}/log_tests.txt"
LOG_FILE_GA="${TESTGEN_GA_OUTPUT_DIR}/log_tests.txt"

if python3 --version &>/dev/null; then
    PYTHON_CMD="python3"
elif python --version &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Neither python3 nor python is properly configured on this system." >&2
    exit 1
fi

echo "Using $PYTHON_CMD to run scripts."


echo "Generating instrumented files."
output=$($PYTHON_CMD instrumentor.py "$ORIGINAL_INPUTS_DIR" "$INSTRUMENTED_INPUTS_DIR" 2>&1) || {
    echo "Error: Failed to generate instrumented files:" >&2
    echo "$output" >&2
    exit 1
}

echo "Generating tests using fuzzer."
output=$($PYTHON_CMD testgen_random.py "$INSTRUMENTED_INPUTS_DIR" "$ORIGINAL_INPUTS_DIR" "$TESTGEN_FUZZ_OUTPUT_DIR" 2>&1) || {
    echo "Error: Failed to generate tests:" >&2
    echo "$output" >&2
    exit 1
}

echo "Generating tests using genetic algorithm."
output=$($PYTHON_CMD testgen_random.py "$INSTRUMENTED_INPUTS_DIR" "$ORIGINAL_INPUTS_DIR" "$TESTGEN_GA_OUTPUT_DIR" 2>&1) || {
    echo "Error: Failed to generate tests:" >&2
    echo "$output" >&2
    exit 1
}

> "$LOG_FILE_FUZZ"
> "$LOG_FILE_GA"

for test_file in "$TESTGEN_FUZZ_OUTPUT_DIR"/*_tests.py; do
    echo "Running tests in $test_file"
    echo "$test_file tests:" >> "$LOG_FILE_FUZZ"
    
    output=$($PYTHON_CMD -m unittest "$test_file" 2>&1) || {
        echo "Error: Tests in $test_file failed:" >&2
        echo "$output" >&2
        exit 1
    }

    echo "$output" >> "$LOG_FILE_FUZZ"
    echo -e "\n\n" >> "$LOG_FILE_FUZZ"
done

for test_file in "$TESTGEN_GA_OUTPUT_DIR"/*_tests.py; do
    echo "Running tests in $test_file"
    echo "$test_file tests:" >> "$LOG_FILE_GA"
    
    output=$($PYTHON_CMD -m unittest "$test_file" 2>&1) || {
        echo "Error: Tests in $test_file failed:" >&2
        echo "$output" >&2
        exit 1
    }

    echo "$output" >> "$LOG_FILE_GA"
    echo -e "\n\n" >> "$LOG_FILE_GA"
done


echo "Test generation and evaluation complete."
echo "Logs are stored in $LOG_FILE_FUZZ and $LOG_FILE_GA."
