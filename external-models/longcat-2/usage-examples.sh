#!/bin/bash

# LongCat-2 Usage Examples for Grok Build CLI

set -e

API_KEY="${LONGCAT_API_KEY:-}"
BASE_URL_OPENAI="https://api.longcat.chat/openai/v1/chat/completions"
BASE_URL_ANTHROPIC="https://api.longcat.chat/anthropic/v1/messages"

if [ -z "$API_KEY" ]; then
    echo "Error: LONGCAT_API_KEY environment variable is not set"
    exit 1
fi

echo "=== LongCat-2 Usage Examples ==="
echo ""

# Example 1: Code Generation
echo "1. Code Generation Example"
echo "Prompt: Write a simple Python function to calculate factorial"
response1=$(curl -s -X POST "$BASE_URL_OPENAI" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LongCat-2.0",
    "messages": [{"role": "user", "content": "Write a simple Python function to calculate factorial"}],
    "max_tokens": 200
  }')

echo "Response:"
echo "$response1" | grep -o '"content":"[^"]*"' | cut -d'"' -f4 | sed 's/\\"/"/g'
echo ""

# Example 2: Code Analysis
echo "2. Code Analysis Example"
echo "Prompt: Analyze this Python code for potential issues: def add(a, b): return a + b"
response2=$(curl -s -X POST "$BASE_URL_OPENAI" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LongCat-2.0",
    "messages": [{"role": "user", "content": "Analyze this Python code for potential issues: def add(a, b): return a + b"}],
    "max_tokens": 150
  }')

echo "Response:"
echo "$response2" | grep -o '"content":"[^"]*"' | cut -d'"' -f4 | sed 's/\\"/"/g'
echo ""

# Example 3: Documentation Generation
echo "3. Documentation Generation Example"
echo "Prompt: Generate documentation for a REST API endpoint"
response3=$(curl -s -X POST "$BASE_URL_ANTHROPIC" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "LongCat-2.0",
    "max_tokens": 200,
    "messages": [{"role": "user", "content": "Generate documentation for a REST API endpoint that creates a user"}]
  }')

echo "Response:"
echo "$response3" | grep -o '"text":"[^"]*"' | cut -d'"' -f4 | sed 's/\\"/"/g'
echo ""

# Example 4: Debugging Assistance
echo "4. Debugging Assistance Example"
echo "Prompt: My Python script is giving a TypeError. How can I debug this?"
response4=$(curl -s -X POST "$BASE_URL_OPENAI" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LongCat-2.0",
    "messages": [{"role": "user", "content": "My Python script is giving a TypeError. How can I debug this?"}],
    "max_tokens": 150
  }')

echo "Response:"
echo "$response4" | grep -o '"content":"[^"]*"' | cut -d'"' -f4 | sed 's/\\"/"/g'
echo ""

echo "=== Usage Examples Complete ==="