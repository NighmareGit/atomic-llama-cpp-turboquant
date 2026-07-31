#!/bin/bash

# Simple LongCat-2 Test Script

echo "=== LongCat-2 Integration Test ==="
echo "API Key: ${LONGCAT_API_KEY:0:10}..."
echo ""

# Test OpenAI format
echo "Testing OpenAI API format..."
response1=$(curl -s -X POST https://api.longcat.chat/openai/v1/chat/completions \
  -H "Authorization: Bearer $LONGCAT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LongCat-2.0",
    "messages": [{"role": "user", "content": "Hello, please introduce yourself briefly."}],
    "max_tokens": 50
  }')

if echo "$response1" | grep -q "error"; then
    echo "❌ OpenAI format failed"
    echo "$response1"
else
    echo "✅ OpenAI format passed"
    id1=$(echo "$response1" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo "   Response ID: $id1"
fi

echo ""

# Test Anthropic format
echo "Testing Anthropic API format..."
response2=$(curl -s -X POST https://api.longcat.chat/anthropic/v1/messages \
  -H "Authorization: Bearer $LONGCAT_API_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "LongCat-2.0",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Hello, please introduce yourself briefly."}]
  }')

if echo "$response2" | grep -q "error"; then
    echo "❌ Anthropic format failed"
    echo "$response2"
else
    echo "✅ Anthropic format passed"
    id2=$(echo "$response2" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo "   Response ID: $id2"
fi

echo ""
echo "=== Test Complete ==="