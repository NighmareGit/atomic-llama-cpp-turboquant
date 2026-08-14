# LongCat-2 Integration for Grok Build CLI

This directory contains configuration and scripts for using LongCat-2 model with Grok Build CLI.

## Overview

LongCat-2 is a high-performance Agentic model available through the LongCat API platform. It supports both OpenAI and Anthropic API formats.

## API Configuration

### API Endpoints
- **OpenAI Format**: `https://api.longcat.chat/openai/v1/chat/completions`
- **Anthropic Format**: `https://api.longcat.chat/anthropic/v1/messages`

### Model Information
- **Model Name**: LongCat-2.0
- **Max Context**: 1M tokens
- **Max Output**: 128K tokens
- **Context Window**: 1M tokens

## Environment Setup

The API key is already configured in the environment:
```bash
echo $LONGCAT_API_KEY
# Output: ak_2Hv4z80qk9yk27d0DI53e12u47d6d
```

## Usage Examples

### 1. Direct cURL Usage (OpenAI Format)

```bash
curl -X POST https://api.longcat.chat/openai/v1/chat/completions \
  -H "Authorization: Bearer $LONGCAT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LongCat-2.0",
    "messages": [{"role": "user", "content": "Hello, please introduce yourself briefly."}],
    "max_tokens": 100
  }'
```

### 2. Direct cURL Usage (Anthropic Format)

```bash
curl -X POST https://api.longcat.chat/anthropic/v1/messages \
  -H "Authorization: Bearer $LONGCAT_API_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "LongCat-2.0",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello, please introduce yourself briefly."}]
  }'
```

### 3. Python Integration (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_LONGCAT_API_KEY",
    base_url="https://api.longcat.chat/openai"
)

response = client.chat.completions.create(
    model="LongCat-2.0",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### 4. Python Integration (Anthropic SDK)

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="YOUR_LONGCAT_API_KEY",
    base_url="https://api.longcat.chat/anthropic"
)

response = client.messages.create(
    model="LongCat-2.0",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.content[0].text)
```

## Grok Build CLI Integration

### Configuration File

Create a Grok Build CLI configuration file at `~/.grok/config/longcat-2.json`:

```json
{
  "provider": "longcat",
  "model": "LongCat-2.0",
  "api_key": "${LONGCAT_API_KEY}",
  "base_url": "https://api.longcat.chat/openai",
  "max_tokens": 4000,
  "temperature": 0.7
}
```

### Grok Build CLI Command

```bash
# Using OpenAI format
grok build --model longcat-2.0 --provider longcat \
  --prompt "Your prompt here" \
  --max-tokens 1000

# Using Anthropic format
grok build --model longcat-2.0 --provider longcat \
  --prompt "Your prompt here" \
  --max-tokens 1000 \
  --api-format anthropic
```

## Smoke Test Results

✅ **OpenAI Format**: Successfully tested
- Response ID: 1dbe4449d3234d6291a452dd753a19c3
- Model: LongCat-2.0
- Tokens Used: 114 total (14 prompt, 100 completion)
- Status: Working

✅ **Anthropic Format**: Successfully tested  
- Response ID: 52a5891fbac84fe08b3d6e37b0dc04fe
- Model: LongCat-2.0
- Tokens Used: 96 total (14 input, 82 output)
- Status: Working

## Rate Limiting

- **Single Request Limit**: 1M token context window
- **Maximum Output**: 128K tokens per request
- **Rate Limit Response**: HTTP 429 with retry_after seconds
- **Recommendation**: Implement exponential backoff retry mechanism

## Error Handling

Common error responses:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Request rate limit exceeded, please try again later",
    "type": "rate_limit_error",
    "retry_after": 60
  }
}
```

## API Documentation

For more detailed documentation, visit: https://longcat.chat/platform/docs/