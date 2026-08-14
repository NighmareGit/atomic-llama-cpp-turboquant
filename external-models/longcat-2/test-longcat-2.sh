#!/bin/bash

# LongCat-2 Integration Test Script
# This script performs smoke tests for both OpenAI and Anthropic API formats

set -e

# Configuration
API_KEY="${LONGCAT_API_KEY:-}"
BASE_URL_OPENAI="https://api.longcat.chat/openai/v1/chat/completions"
BASE_URL_ANTHROPIC="https://api.longcat.chat/anthropic/v1/messages"

# Check if API key is set
if [ -z "$API_KEY" ]; then
    echo "Error: LONGCAT_API_KEY environment variable is not set"
    exit 1
fi

# Test prompts
PROMPT="Hello, please introduce yourself briefly."
MODEL="LongCat-2.0"

echo "=== LongCat-2 Integration Test ==="
echo "API Key: ${API_KEY:0:10}..."
echo "Model: $MODEL"
echo ""

# Function to test API endpoint
test_endpoint() {
    local endpoint="$1"
    local format="$2"
    local headers="$3"
    
    echo "Testing $format API format..."
    echo "Endpoint: $endpoint"
    
    # Make the API call
    response=$(curl -s -X POST "$endpoint" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        $headers \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 50
        }")
    
    # Check response
    if echo "$response" | grep -q "error"; then
        echo "❌ Error in $format format:"
        echo "$response"
        return 1
    else
        echo "✅ $format format test passed"
        
        # Extract and display key info
        if echo "$response" | grep -q "id"; then
            local id=$(echo "$response" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
            echo "   Response ID: $id"
        fi
        
        # Handle different response formats
        if echo "$response" | grep -q "total_tokens"; then
            local tokens=$(echo "$response" | grep -o '"total_tokens":[0-9]*' | cut -d':' -f2)
            echo "   Tokens Used: $tokens"
        elif echo "$response" | grep -q "input_tokens"; then
            local input_tokens=$(echo "$response" | grep -o '"input_tokens":[0-9]*' | cut -d':' -f2)
            local output_tokens=$(echo "$response" | grep -o '"output_tokens":[0-9]*' | cut -d':' -f2)
            echo "   Tokens Used: $input_tokens input, $output_tokens output"
        fi
        
        # Extract and show a snippet of the response
        if echo "$response" | grep -q "content"; then
            if echo "$response" | grep -q "text"; then
                # Anthropic format
                local content=$(echo "$response" | grep -o '"text":"[^"]*"' | cut -d'"' -f4 | head -c 100)
                echo "   Response: $content..."
            else
                # OpenAI format
                local content=$(echo "$response" | grep -o '"content":"[^"]*"' | cut -d'"' -f4 | head -c 100)
                echo "   Response: $content..."
            fi
        fi
        
        echo ""
    fi
}

# Test OpenAI format
test_endpoint "$BASE_URL_OPENAI" "OpenAI" ""

# Test Anthropic format
test_endpoint "$BASE_URL_ANTHROPIC" "Anthropic" "-H \"anthropic-version: 2023-06-01\""

echo "=== All Tests Completed ==="
echo "LongCat-2 integration is ready for use with Grok Build CLI!"