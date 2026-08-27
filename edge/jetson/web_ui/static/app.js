const frameImage = document.getElementById("frame-image");
const frameOverlay = document.getElementById("frame-overlay");
const frameEmpty = document.getElementById("frame-empty");
const frameIdEl = document.getElementById("frame-id");
const frameTimestampEl = document.getElementById("frame-timestamp");
const detectionCountEl = document.getElementById("detection-count");
const detectionListEl = document.getElementById("detection-list");
const connectionStatusEl = document.getElementById("connection-status");
const statusDotEl = document.getElementById("status-dot");
const streamStateEl = document.getElementById("stream-state");

const overlayContext = frameOverlay.getContext("2d");

let socket = null;
let reconnectTimer = null;
let reconnectAttempt = 0;
let latestFrame = null;
let imageReady = false;
let pendingFrame = null;
let pendingLoadToken = 0;

function setConnectionState(state, label) {
  statusDotEl.dataset.state = state;
  connectionStatusEl.textContent = label;
}

function formatTimestamp(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium"
  }).format(date);
}

function resizeOverlay() {
  const bounds = frameOverlay.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));

  if (frameOverlay.width !== width || frameOverlay.height !== height) {
    frameOverlay.width = width;
    frameOverlay.height = height;
  }

  drawOverlay();
}

function hideStage(message) {
  imageReady = false;
  frameImage.style.display = "none";
  frameImage.removeAttribute("src");
  frameEmpty.style.display = "grid";
  frameEmpty.textContent = message;
  drawOverlay();
}

function drawOverlay() {
  overlayContext.clearRect(0, 0, frameOverlay.width, frameOverlay.height);

  if (!latestFrame || !imageReady || !Array.isArray(latestFrame.detections)) {
    return;
  }

  const sourceWidth = frameImage.naturalWidth || frameOverlay.width;
  const sourceHeight = frameImage.naturalHeight || frameOverlay.height;
  const scale = Math.max(frameOverlay.width / sourceWidth, frameOverlay.height / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const offsetX = (frameOverlay.width - renderedWidth) / 2;
  const offsetY = (frameOverlay.height - renderedHeight) / 2;

  overlayContext.lineWidth = 3;
  overlayContext.font = "600 16px 'Avenir Next', 'Segoe UI', sans-serif";
  overlayContext.strokeStyle = "#62f5c6";
  overlayContext.fillStyle = "#62f5c6";

  latestFrame.detections.forEach((detection) => {
    const box = detection.bbox;
    if (!box) {
      return;
    }

    const x = offsetX + box.x * scale;
    const y = offsetY + box.y * scale;
    const width = box.width * scale;
    const height = box.height * scale;

    overlayContext.strokeRect(x, y, width, height);

    const label = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;
    const textWidth = overlayContext.measureText(label).width;
    const textY = Math.max(20, y - 10);

    overlayContext.fillRect(x, textY - 18, textWidth + 14, 24);
    overlayContext.fillStyle = "#061018";
    overlayContext.fillText(label, x + 7, textY);
    overlayContext.fillStyle = "#62f5c6";
  });
}

function renderDetections(detections) {
  detectionCountEl.textContent = `${detections.length} active`;
  detectionListEl.replaceChildren();

  if (detections.length === 0) {
    const item = document.createElement("li");
    item.className = "detection-list__empty";
    item.textContent = "No detections in the latest frame.";
    detectionListEl.appendChild(item);
    return;
  }

  detections.forEach((detection, index) => {
    const item = document.createElement("li");
    const title = document.createElement("div");
    const label = document.createElement("strong");
    const scoreValue = document.createElement("span");
    const meta = document.createElement("div");
    const position = document.createElement("span");
    const size = document.createElement("span");
    const score = `${(detection.score * 100).toFixed(1)}%`;
    const box = detection.bbox || { x: 0, y: 0, width: 0, height: 0 };

    title.className = "detection-title";
    meta.className = "detection-meta";
    label.textContent = `${index + 1}. ${detection.label}`;
    scoreValue.textContent = score;
    position.textContent = `x:${box.x} y:${box.y}`;
    size.textContent = `${box.width}×${box.height}`;

    title.append(label, scoreValue);
    meta.append(position, size);
    item.append(title, meta);

    detectionListEl.appendChild(item);
  });
}

function applyFrame(frame, options = {}) {
  const detections = Array.isArray(frame.detections) ? frame.detections : [];

  latestFrame = frame;
  frameIdEl.textContent = frame.frame_id || "Unknown";
  frameTimestampEl.textContent = formatTimestamp(frame.timestamp || "");
  streamStateEl.textContent = detections.length > 0 ? "Objects tracked" : "No objects tracked";
  renderDetections(detections);

  if (options.showImage) {
    frameImage.style.display = "block";
    frameEmpty.style.display = "none";
  } else {
    hideStage(options.emptyMessage || "Frame unavailable.");
  }
}

function renderFrame(frame) {
  pendingLoadToken += 1;
  const loadToken = pendingLoadToken;
  pendingFrame = frame;

  if (frame.image_payload) {
    hideStage("Loading latest frame…");

    frameImage.onload = () => {
      if (loadToken !== pendingLoadToken || pendingFrame !== frame) {
        return;
      }

      imageReady = true;
      applyFrame(frame, { showImage: true });
      resizeOverlay();
    };

    frameImage.onerror = () => {
      if (loadToken !== pendingLoadToken || pendingFrame !== frame) {
        return;
      }

      applyFrame(frame, {
        showImage: false,
        emptyMessage: "Latest frame image could not be loaded."
      });
    };

    frameImage.src = frame.image_payload;
  } else {
    applyFrame(frame, {
      showImage: false,
      emptyMessage: "Latest frame has no image payload."
    });
    drawOverlay();
  }
}

function scheduleReconnect() {
  if (reconnectTimer) {
    return;
  }

  reconnectAttempt += 1;
  const delayMs = Math.min(1000 * 2 ** Math.min(reconnectAttempt - 1, 4), 10000);
  setConnectionState("reconnecting", `Reconnecting in ${Math.round(delayMs / 1000)}s`);

  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, delayMs);
}

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socketUrl = `${protocol}//${window.location.host}/ws`;

  setConnectionState("reconnecting", "Connecting");
  socket = new WebSocket(socketUrl);

  socket.addEventListener("open", () => {
    reconnectAttempt = 0;
    setConnectionState("connected", "Live");
  });

  socket.addEventListener("message", (event) => {
    try {
      const frame = JSON.parse(event.data);
      renderFrame(frame);
    } catch (error) {
      console.error("failed to parse frame payload", error);
    }
  });

  socket.addEventListener("close", () => {
    setConnectionState("disconnected", "Disconnected");
    scheduleReconnect();
  });

  socket.addEventListener("error", () => {
    if (socket) {
      socket.close();
    }
  });
}

window.addEventListener("resize", resizeOverlay);

resizeOverlay();
connect();
