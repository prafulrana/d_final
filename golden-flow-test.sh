#!/bin/bash
echo "🏆 GOLDEN FLOW: Starting complete waterfall orchestration..."
echo "🔌 Connecting to WebSocket for live updates..."

# Start WebSocket client using node
node -e "
const WebSocket = require('ws');
const axios = require('axios');

const ws = new WebSocket('ws://localhost:3001');

ws.on('open', async () => {
  console.log('🔌 Connected to WebSocket');

  // Start golden flow
  try {
    await axios.post('http://localhost:3001/api/test');
  } catch (e) {
    console.log('Golden flow started...');
  }
});

ws.on('message', (data) => {
  const message = JSON.parse(data);
  console.log(message.text);

  if (message.type === 'complete') {
    console.log('');
    console.log('✅ GOLDEN FLOW COMPLETED!');
    console.log('📁 Output file:', message.outputFile);
    process.exit(0);
  }
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error.message);
  process.exit(1);
});

// Timeout after 5 minutes
setTimeout(() => {
  console.log('❌ Timeout after 5 minutes');
  process.exit(1);
}, 300000);
"