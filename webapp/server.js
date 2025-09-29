const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
// Removed: exec and execAsync - using Docker socket API only
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const PORT = 3001;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Docker API client via unix socket
const dockerAPI = axios.create({
  socketPath: '/var/run/docker.sock',
  baseURL: 'http://localhost'
});

// WebSocket broadcast function
function broadcast(message) {
  const data = JSON.stringify(message);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  });
  console.log(message.text);
}

// CONTAINER CONFIGS (on-demand only)
const CONTAINER_CONFIGS = {
  mediamtx: {
    Image: 'bluenviron/mediamtx:latest',
    name: 'mediamtx-master',
    HostConfig: {
      NetworkMode: 'host',
      RestartPolicy: { Name: 'unless-stopped' }
    }
  },

  ffmpegCam1: {
    Image: 'jrottenberg/ffmpeg:4.1-alpine',
    name: 'ffmpeg-cam1-master',
    Cmd: [
      '-re', '-stream_loop', '-1',
      '-i', '/samples/sample_1080p_h264.mp4',
      '-c', 'copy', '-f', 'rtsp',
      'rtsp://localhost:8554/cam1'
    ],
    HostConfig: {
      NetworkMode: 'host',
      Binds: ['/opt/nvidia/deepstream/deepstream/samples/streams:/samples:ro'],
      RestartPolicy: { Name: 'unless-stopped' }
    }
  },

  ffmpegCam2: {
    Image: 'jrottenberg/ffmpeg:4.1-alpine',
    name: 'ffmpeg-cam2-master',
    Cmd: [
      '-re', '-stream_loop', '-1',
      '-i', '/samples/sample_1080p_h264.mp4',
      '-c', 'copy', '-f', 'rtsp',
      'rtsp://localhost:8554/cam2'
    ],
    HostConfig: {
      NetworkMode: 'host',
      Binds: ['/opt/nvidia/deepstream/deepstream/samples/streams:/samples:ro'],
      RestartPolicy: { Name: 'unless-stopped' }
    }
  },

  deepstream: {
    Image: 'pyservicemaker-hello:latest',
    name: 'deepstream-master',
    HostConfig: {
      NetworkMode: 'host',
      DeviceRequests: [{ Driver: 'nvidia', Count: -1, Capabilities: [['gpu']] }],
      RestartPolicy: { Name: 'unless-stopped' }
    }
  }
};

// CONTAINER MANAGEMENT FUNCTIONS
async function cleanupContainer(containerName) {
  try {
    const containers = await dockerAPI.get('/containers/json', {
      params: { all: true, filters: JSON.stringify({ name: [containerName] }) }
    });

    if (containers.data.length > 0) {
      const container = containers.data[0];
      await dockerAPI.post(`/containers/${container.Id}/kill`).catch(() => {});
      await dockerAPI.delete(`/containers/${container.Id}`).catch(() => {});
      console.log(`🧹 Cleaned up container: ${containerName}`);
    }
  } catch (error) {
    // Container cleanup failures are non-fatal
  }
}

async function getContainerStatus(containerName) {
  try {
    const containers = await dockerAPI.get('/containers/json', {
      params: { all: true, filters: JSON.stringify({ name: [containerName] }) }
    });

    if (containers.data.length > 0) {
      const container = containers.data[0];
      return {
        exists: true,
        running: container.State === 'running',
        id: container.Id
      };
    }
    return { exists: false, running: false, id: null };
  } catch (error) {
    return { exists: false, running: false, id: null };
  }
}

async function startContainer(configKey) {
  const config = CONTAINER_CONFIGS[configKey];
  console.log(`🚀 Starting ${config.name}`);

  try {
    // Remove existing container if any
    const existing = await getContainerStatus(config.name);
    if (existing.exists) {
      await dockerAPI.post(`/containers/${existing.id}/kill`);
      await dockerAPI.delete(`/containers/${existing.id}`);
    }

    // Create and start new container
    const response = await dockerAPI.post('/containers/create', config, {
      params: { name: config.name }
    });
    await dockerAPI.post(`/containers/${response.data.Id}/start`);
    console.log(`✅ ${config.name} started`);

    return response.data.Id;
  } catch (error) {
    console.error(`❌ Failed to start ${config.name}:`, error.message);
    throw error;
  }
}

async function waitForMediaMTXReady() {
  const net = require('net');
  return new Promise((resolve) => {
    const checkPort = () => {
      const socket = new net.Socket();
      socket.setTimeout(1000);
      socket.on('connect', () => {
        console.log(`✅ MediaMTX RTSP port ready: 8554`);
        socket.destroy();
        resolve();
      });
      socket.on('timeout', () => {
        socket.destroy();
        setTimeout(checkPort, 1000);
      });
      socket.on('error', () => {
        socket.destroy();
        setTimeout(checkPort, 1000);
      });
      socket.connect(8554, 'localhost');
    };
    checkPort();
  });
}

async function waitForRTSPStreamPublished(streamName) {
  return new Promise((resolve) => {
    const checkLogs = async () => {
      try {
        const logs = await dockerAPI.get('/containers/mediamtx-master/logs', {
          params: { stdout: true, stderr: true, tail: 10 }
        });

        if (logs.data.includes(`is publishing to path '${streamName}'`)) {
          console.log(`✅ RTSP stream published: ${streamName}`);
          resolve();
        } else {
          setTimeout(checkLogs, 1000);
        }
      } catch {
        setTimeout(checkLogs, 1000);
      }
    };
    checkLogs();
  });
}

async function waitForDeepStreamUDPOutput(streamId) {
  return new Promise((resolve) => {
    const checkLogs = async () => {
      try {
        const logs = await dockerAPI.get('/containers/deepstream-master/logs', {
          params: { stdout: true, stderr: true, tail: 20 }
        });

        if (logs.data.includes('All RTSP streams available!') || logs.data.includes('pipeline')) {
          console.log(`✅ DeepStream UDP output active for stream ${streamId}`);
          resolve();
        } else {
          setTimeout(checkLogs, 1000);
        }
      } catch {
        setTimeout(checkLogs, 1000);
      }
    };
    checkLogs();
  });
}

// START MEDIAMTX ON NODE.JS STARTUP (ONLY PERSISTENT CONTAINER)
async function startMediaMTX() {
  console.log("🚀 Starting MediaMTX (only persistent container)");
  try {
    await startContainer('mediamtx');
    console.log("✅ MediaMTX ready");
  } catch (error) {
    console.error("❌ MediaMTX startup failed:", error.message);
  }
}

// ON-DEMAND TEST ENDPOINT
app.post('/api/test-stream/:id', async (req, res) => {
  const streamId = req.params.id;
  console.log(`🎯 Testing stream ${streamId} - ON-DEMAND MODE`);

  try {
    // STEP 0: Implicit cleanup first
    await cleanupContainer(`ffmpeg-recorder-${streamId}`);
    await cleanupContainer('ffmpeg-cam1-master');
    await cleanupContainer('ffmpeg-cam2-master');
    await cleanupContainer('deepstream-master');

    // STEP 1: Ensure MediaMTX is ready
    console.log(`📡 Waiting for MediaMTX RTSP port...`);
    await waitForMediaMTXReady();

    // STEP 2: Start FFmpeg feeder
    const feederKey = streamId === '0' ? 'ffmpegCam1' : 'ffmpegCam2';
    const streamName = `cam${parseInt(streamId) + 1}`;
    console.log(`📡 Starting FFmpeg feeder: ${feederKey}`);
    await startContainer(feederKey);

    // STEP 3: Wait for RTSP stream to be published
    console.log(`📡 Waiting for RTSP stream published: ${streamName}`);
    await waitForRTSPStreamPublished(streamName);

    // STEP 4: Start DeepStream pipeline
    console.log(`🧠 Starting DeepStream AI pipeline`);
    await startContainer('deepstream');

    // STEP 5: Wait for DeepStream UDP output
    console.log(`🧠 Waiting for DeepStream UDP output...`);
    await waitForDeepStreamUDPOutput(streamId);

    // STEP 4: Start temporary recorder
    console.log(`📹 Starting recorder for stream ${streamId}`);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const recorderContainerName = `ffmpeg-recorder-${streamId}`;
    const outputDir = `/home/prafulrana/d/outputs`;
    const outputFile = `/outputs/stream_${streamId}_${timestamp}.mp4`;

    const recorderConfig = {
      Image: 'jrottenberg/ffmpeg:4.1-alpine',
      name: recorderContainerName,
      Cmd: [
        '-i', `udp://224.224.255.255:500${streamId}`,
        '-t', '10',
        '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-movflags', '+faststart',
        '-y', outputFile
      ],
      HostConfig: {
        NetworkMode: 'host',
        Binds: ['/home/prafulrana/d/outputs:/outputs'],
        AutoRemove: true
      }
    };

    const recorderResponse = await dockerAPI.post('/containers/create', recorderConfig, {
      params: { name: recorderContainerName }
    });

    await dockerAPI.post(`/containers/${recorderResponse.data.Id}/start`);
    console.log(`✅ Started recorder for stream ${streamId}`);

    // STEP 7: Wait for recording to complete (10 seconds)
    console.log(`📹 Waiting for 10 seconds of recording...`);
    await new Promise((resolve) => {
      const checkRecording = async () => {
        try {
          const containerStatus = await dockerAPI.get(`/containers/${recorderResponse.data.Id}/json`);
          if (containerStatus.data.State.Status === 'exited') {
            console.log(`✅ Recording completed for stream ${streamId}`);
            resolve();
          } else {
            setTimeout(checkRecording, 1000);
          }
        } catch {
          setTimeout(checkRecording, 1000);
        }
      };
      checkRecording();
    });

    // STEP 8: Graceful shutdown of entire chain
    console.log(`🧹 Graceful shutdown of sender/receiver chain...`);
    await cleanupContainer(`ffmpeg-recorder-${streamId}`);
    await cleanupContainer('deepstream-master');
    await cleanupContainer('ffmpeg-cam1-master');
    await cleanupContainer('ffmpeg-cam2-master');

    res.json({
      status: 'completed',
      message: `Recorded stream ${streamId} with AI detection`,
      outputFile: outputFile.replace('/outputs/', ''),
      duration: 10
    });

  } catch (error) {
    console.error(`💀 Error testing stream ${streamId}:`, error.message);
    res.status(500).json({ error: error.message });
  }
});

// Stream video endpoint
app.get('/api/stream/:id', (req, res) => {
  const streamId = req.params.id;
  const outputDir = '/home/prafulrana/d/outputs';

  // Find latest file for this stream
  const fs = require('fs');
  if (!fs.existsSync(outputDir)) {
    return res.status(404).json({ error: 'No recordings found' });
  }

  const files = fs.readdirSync(outputDir)
    .filter(f => f.startsWith(`stream_${streamId}_`))
    .sort().reverse();

  if (files.length === 0) {
    return res.status(404).json({ error: 'Stream not ready yet' });
  }

  const videoFile = `${outputDir}/${files[0]}`;

  if (!fs.existsSync(videoFile)) {
    return res.status(404).json({ error: 'Stream not ready yet' });
  }

  const stat = fs.statSync(videoFile);
  const fileSize = stat.size;
  const range = req.headers.range;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Range');
  res.setHeader('Cache-Control', 'no-cache');

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : Math.min(start + 1024 * 1024, fileSize - 1);
    const chunksize = (end - start) + 1;

    const file = fs.createReadStream(videoFile, { start, end });
    const head = {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': 'video/mp4',
      'X-Content-Type-Options': 'nosniff'
    };
    res.writeHead(206, head);
    file.pipe(res);
  } else {
    const head = {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
      'Accept-Ranges': 'bytes',
      'X-Content-Type-Options': 'nosniff'
    };
    res.writeHead(200, head);
    fs.createReadStream(videoFile).pipe(res);
  }
});

// Cleanup endpoint - API-driven only
app.post('/api/cleanup', async (req, res) => {
  console.log('🧹 Starting API-driven cleanup...');
  try {
    const containerNames = [
      'ffmpeg-recorder-0', 'ffmpeg-recorder-1',
      'ffmpeg-cam1-master', 'ffmpeg-cam2-master',
      'deepstream-master'
    ];

    for (const name of containerNames) {
      await cleanupContainer(name);
    }

    res.json({ status: 'cleaned', containers: containerNames });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Container logs endpoint - API-driven debugging
app.get('/api/logs/:containerName', async (req, res) => {
  try {
    const containerName = req.params.containerName;
    const containers = await dockerAPI.get('/containers/json', {
      params: { all: true, filters: JSON.stringify({ name: [containerName] }) }
    });

    if (containers.data.length === 0) {
      return res.status(404).json({ error: 'Container not found' });
    }

    const containerId = containers.data[0].Id;
    const logs = await dockerAPI.get(`/containers/${containerId}/logs`, {
      params: { stdout: true, stderr: true, tail: 50 }
    });

    res.json({ container: containerName, logs: logs.data });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// 🏆 GOLDEN FLOW - Complete waterfall orchestration test with live updates
app.post('/api/test', async (req, res) => {
  broadcast({ type: 'status', text: '🏆 GOLDEN FLOW: Starting complete waterfall orchestration...' });

  // Return immediately, process continues with websocket updates
  res.json({ status: 'started', message: 'Golden flow started, connect to websocket for live updates' });

  try {
    // Use stream 0 for golden flow
    const streamId = '0';
    const streamName = 'cam1';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputDir = `/home/prafulrana/d/outputs`;
    const outputFile = `/outputs/golden_flow_${timestamp}.mp4`;

    broadcast({ type: 'step', text: '🧹 STEP 1: Complete cleanup...' });
    await cleanupContainer(`ffmpeg-recorder-${streamId}`);
    await cleanupContainer('ffmpeg-cam1-master');
    await cleanupContainer('ffmpeg-cam2-master');
    await cleanupContainer('deepstream-master');

    broadcast({ type: 'step', text: '📡 STEP 2: Waiting for MediaMTX RTSP port...' });
    await waitForMediaMTXReady();

    broadcast({ type: 'step', text: '📡 STEP 3: Starting FFmpeg feeder...' });
    await startContainer('ffmpegCam1');

    broadcast({ type: 'step', text: '📡 STEP 4: Waiting for RTSP stream published...' });
    await waitForRTSPStreamPublished(streamName);

    broadcast({ type: 'step', text: '🧠 STEP 5: Starting DeepStream AI pipeline...' });
    await startContainer('deepstream');

    broadcast({ type: 'step', text: '🧠 STEP 6: Waiting for DeepStream UDP output...' });
    await waitForDeepStreamUDPOutput(streamId);

    broadcast({ type: 'step', text: '📹 STEP 7: Starting recorder...' });
    const recorderConfig = {
      Image: 'jrottenberg/ffmpeg:4.1-alpine',
      name: `golden-flow-recorder`,
      Cmd: [
        '-i', `udp://224.224.255.255:5000`,
        '-t', '10',
        '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-movflags', '+faststart',
        '-y', outputFile
      ],
      HostConfig: {
        NetworkMode: 'host',
        Binds: ['/home/prafulrana/d/outputs:/outputs'],
        AutoRemove: true
      }
    };

    const recorderResponse = await dockerAPI.post('/containers/create', recorderConfig, {
      params: { name: recorderConfig.name }
    });
    await dockerAPI.post(`/containers/${recorderResponse.data.Id}/start`);

    broadcast({ type: 'step', text: '📹 STEP 8: Waiting for recording completion...' });
    await new Promise((resolve) => {
      const checkRecording = async () => {
        try {
          const containerStatus = await dockerAPI.get(`/containers/${recorderResponse.data.Id}/json`);
          if (containerStatus.data.State.Status === 'exited') {
            broadcast({ type: 'success', text: '✅ Recording completed!' });
            resolve();
          } else {
            setTimeout(checkRecording, 1000);
          }
        } catch {
          setTimeout(checkRecording, 1000);
        }
      };
      checkRecording();
    });

    broadcast({ type: 'step', text: '🧹 STEP 9: Graceful shutdown...' });
    await cleanupContainer('golden-flow-recorder');
    await cleanupContainer('deepstream-master');
    await cleanupContainer('ffmpeg-cam1-master');

    broadcast({
      type: 'complete',
      text: '🏆 GOLDEN FLOW COMPLETED: Full AI detection pipeline test',
      outputFile: outputFile.replace('/outputs/', ''),
      timestamp: timestamp
    });

  } catch (error) {
    console.error('💀 GOLDEN FLOW FAILED:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Health check
app.get('/api/health', async (req, res) => {
  try {
    const statuses = {};
    for (const [key, config] of Object.entries(CONTAINER_CONFIGS)) {
      const status = await getContainerStatus(config.name);
      statuses[key] = status.running;
    }

    res.json({
      status: 'ok',
      containers: statuses,
      mode: 'on-demand'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// SIMPLE STARTUP: Start MediaMTX then web server with WebSocket
startMediaMTX().then(() => {
  server.listen(PORT, () => {
    console.log(`🎯 DeepStream Tester running on port ${PORT}`);
    console.log(`📋 Golden Flow available via: npm test (with live WebSocket updates)`);
    console.log(`🔌 WebSocket server running for live updates`);
  });
});