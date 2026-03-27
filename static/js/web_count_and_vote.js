const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const video = document.getElementById('video');
const videoWrap = document.getElementById('videoWrap');
const page = document.querySelector('.page');
const backendView = document.getElementById('backendView');
const streamView = document.getElementById('streamView');
const voteInfoView = document.getElementById('voteInfoView');
const countView = document.getElementById('countView');
const panelToggleBar = document.getElementById('panelToggleBar');
const panelToggleHandle = document.getElementById('panelToggleHandle');
const panelToggleContent = document.getElementById('panelToggleContent');
const toggleBackendViewInput = document.getElementById('toggleBackendView');
const toggleStreamViewInput = document.getElementById('toggleStreamView');
const toggleVoteInfoViewInput = document.getElementById('toggleVoteInfoView');
const toggleCountViewInput = document.getElementById('toggleCountView');

let state = null;
let regions = [];
let selectedRegionId = null;
let draggingPointIdx = null;
let activePointerId = null;
let pointDragMoved = false;
let draggingRegion = false;
let dragLastX = null;
let dragLastY = null;
let regionsDirty = false;
let regionSaveInFlight = false;
let slotColors = { 1: '#0066cc', 2: '#ffcc00' };
let votingConfigSaveInFlight = false;
let votingConfigSaveQueued = false;
let votingConfigDebounceHandle = null;
let lastVotingConfigSignature = null;
let detectorSaveInFlight = false;
let detectorSaveQueued = false;
let detectorDebounceHandle = null;
let lastDetectorSignature = null;
let canvasDisplayWidth = 1;
let canvasDisplayHeight = 1;
let resizeSyncFrameHandle = null;
let mirrorEnabled = false;
let showDetectionLabel = true;

const MIRROR_VIEW_STORAGE_KEY = 'look_and_detect_mirror_view';
const SHOW_DETECTION_LABEL_STORAGE_KEY = 'look_and_detect_show_detection_label';
const PANEL_VISIBILITY_STORAGE_KEY = 'look_and_detect_panel_visibility';
const UPDATE_INTERVAL_STORAGE_KEY = 'look_and_detect_update_interval';
const STREAM_FEED_URL = '/video_feed';
let streamActive = true;
let updateIntervalMs = parseInt(document.currentScript?.dataset.updateInterval, 10) || 650;
const markdownContentCache = new WeakMap();

const chart = new Chart(document.getElementById('countChart').getContext('2d'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    animation: false,
    responsive: true,
    plugins: { legend: { position: 'bottom' } },
    scales: { x: { display: false }, y: { beginAtZero: true } }
  }
});

function flash(msg) {
  document.getElementById('flash').textContent = msg;
}

function withAlpha(hexColor, alpha = '33') {
  return `${hexColor}${alpha}`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function isImageUrl(url) {
  if (!url) return false;
  const urlStr = String(url).trim();
  
  // Check for http/https URLs
  if (!/^https?:\/\//i.test(urlStr)) {
    return false;
  }
  
  // Check if URL ends with image file extensions (jpg, jpeg, png, webm, gif, svg, webp, bmp)
  // Match before query string or hash
  const pathOnly = urlStr.split(/[?#]/)[0];
  return /\.(jpg|jpeg|png|webm|gif|svg|webp|bmp)$/i.test(pathOnly);
}

function renderInlineMarkdown(line) {
  let html = escapeHtml(line);
  html = html.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  return html;
}

function renderBasicMarkdown(markdownText) {
  const normalized = String(markdownText ?? '').trim();
  if (!normalized) {
    return '';
  }

  let html = normalized;

  // Preserve raw HTML img tags by temporarily replacing them
  const imgTags = [];
  html = html.replace(/<img\s+[^>]*>/gi, (match) => {
    imgTags.push(match);
    return `__IMG_PLACEHOLDER_${imgTags.length - 1}__`;
  });

  const blocks = html.split(/\n\s*\n+/);
  const renderedBlocks = blocks.map((block) => {
    let trimmed = block.trim();

    // Check if this is only a placeholder - restore it as-is
    if (/^__IMG_PLACEHOLDER_\d+__$/.test(trimmed)) {
      const match = trimmed.match(/__IMG_PLACEHOLDER_(\d+)__/);
      if (match) {
        return imgTags[parseInt(match[1])];
      }
    }

    // Check for markdown image syntax ![alt](url)
    const markdownImage = trimmed.match(/^!\[([^\]]*)\]\((https?:\/\/[^\s)]+)\)$/i);
    if (markdownImage && isImageUrl(markdownImage[2])) {
      const alt = escapeHtml(markdownImage[1] || 'Question image');
      const src = markdownImage[2];
      return `<img class="markdown-image" src="${src}" alt="${alt}" loading="lazy" />`;
    }

    // Check if entire block is a plain image URL
    if (isImageUrl(trimmed)) {
      return `<img class="markdown-image" src="${trimmed}" alt="Question image" loading="lazy" />`;
    }

    // Replace image URLs within text with img tags, preserving surrounding text
    // This handles cases like: "Text before https://example.com/image.jpg text after"
    trimmed = trimmed.replace(/(https?:\/\/[^\s<>"{}|\\^`\[\]]*\.(?:jpg|jpeg|png|webm|gif|svg|webp|bmp)(?:\?[^\s<>"{}|\\^`\[\]]*)?)/gi, 
      (url) => `<img class="markdown-image" src="${url}" alt="Image" style="max-width:100%; display:inline;" />`
    );

    // Render as inline text with markdown formatting
    // First escape HTML, then do markdown replacements, then restore placeholders
    let result = trimmed
      .split('\n')
      .map((line) => {
        // Keep image tags/placeholders as-is, but still render markdown around them.
        const segments = line.split(/(<img\b[^>]*>|__IMG_PLACEHOLDER_\d+__)/g);
        return segments
          .map((segment) => {
            if (/^(<img\b[^>]*>|__IMG_PLACEHOLDER_\d+__)$/.test(segment)) {
              return segment;
            }
            return renderInlineMarkdown(segment);
          })
          .join('');
      })
      .join('<br>');

    // Restore any img tag placeholders in the result
    result = result.replace(/__IMG_PLACEHOLDER_(\d+)__/g, (match, index) => {
      return imgTags[parseInt(index)];
    });

    return result;
  });

  return renderedBlocks.join('<br>');
}

function setMarkdownContent(elementId, markdownText, fallbackText = '') {
  const element = document.getElementById(elementId);
  if (!element) {
    return;
  }

  const source = String(markdownText ?? '').trim();
  if (!source) {
    const emptyKey = `text:${fallbackText}`;
    if (markdownContentCache.get(element) === emptyKey) {
      return;
    }
    element.textContent = fallbackText;
    markdownContentCache.set(element, emptyKey);
    return;
  }

  const html = renderBasicMarkdown(source);
  const rendered = html || escapeHtml(fallbackText);
  const htmlKey = `html:${rendered}`;
  if (markdownContentCache.get(element) === htmlKey) {
    return;
  }
  element.innerHTML = rendered;
  markdownContentCache.set(element, htmlKey);
}

function getSlotColor(slot) {
  const normalizedSlot = Number(slot) === 2 ? 2 : 1;
  return slotColors[normalizedSlot] || (normalizedSlot === 1 ? '#0066cc' : '#ffcc00');
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  const wrapRect = videoWrap.getBoundingClientRect();
  const nextDisplayWidth = Math.max(1, Math.round(rect.width));
  const nextDisplayHeight = Math.max(1, Math.round(rect.height));
  const devicePixelRatio = Math.max(1, window.devicePixelRatio || 1);

  canvasDisplayWidth = nextDisplayWidth;
  canvasDisplayHeight = nextDisplayHeight;
  canvas.width = Math.max(1, Math.round(nextDisplayWidth * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(nextDisplayHeight * devicePixelRatio));
  canvas.style.width = nextDisplayWidth + 'px';
  canvas.style.height = nextDisplayHeight + 'px';
  canvas.style.left = (rect.left - wrapRect.left) + 'px';
  canvas.style.top = (rect.top - wrapRect.top) + 'px';
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  drawOverlay();
}

function getCanvasDisplayWidth() {
  return Math.max(1, canvasDisplayWidth || Math.round(canvas.getBoundingClientRect().width) || 1);
}

function getCanvasDisplayHeight() {
  return Math.max(1, canvasDisplayHeight || Math.round(canvas.getBoundingClientRect().height) || 1);
}

function scheduleResizeSync(frameCount = 10) {
  if (resizeSyncFrameHandle !== null) {
    cancelAnimationFrame(resizeSyncFrameHandle);
    resizeSyncFrameHandle = null;
  }

  let remainingFrames = Math.max(1, frameCount);
  const tick = () => {
    resizeCanvas();
    remainingFrames -= 1;
    if (remainingFrames > 0) {
      resizeSyncFrameHandle = requestAnimationFrame(tick);
      return;
    }
    resizeSyncFrameHandle = null;
  };

  resizeSyncFrameHandle = requestAnimationFrame(tick);
}

function hasFrameState() {
  return !!(
    state &&
    state.frame &&
    Number.isFinite(state.frame.width) &&
    Number.isFinite(state.frame.height) &&
    state.frame.width > 0 &&
    state.frame.height > 0
  );
}

function toDisplayPoint(p) {
  if (!hasFrameState()) return p;
  const sx = getCanvasDisplayWidth() / state.frame.width;
  const sy = getCanvasDisplayHeight() / state.frame.height;
  const mappedX = p[0] * sx;
  return [mirrorEnabled ? (getCanvasDisplayWidth() - mappedX) : mappedX, p[1] * sy];
}

function toFramePoint(x, y) {
  if (!hasFrameState()) return [Math.round(x), Math.round(y)];
  const sx = state.frame.width / getCanvasDisplayWidth();
  const sy = state.frame.height / getCanvasDisplayHeight();
  const mappedX = mirrorEnabled ? (getCanvasDisplayWidth() - x) : x;
  return [Math.round(mappedX * sx), Math.round(y * sy)];
}

function getCanvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

function setMirrorEnabled(enabled, { persist = true } = {}) {
  mirrorEnabled = !!enabled;
  videoWrap.classList.toggle('is-mirrored', mirrorEnabled);

  const mirrorViewInput = document.getElementById('mirrorView');
  if (mirrorViewInput && mirrorViewInput.checked !== mirrorEnabled) {
    mirrorViewInput.checked = mirrorEnabled;
  }

  if (persist) {
    try {
      localStorage.setItem(MIRROR_VIEW_STORAGE_KEY, mirrorEnabled ? '1' : '0');
    } catch (_err) {
      // Ignore storage failures and keep runtime-only behavior.
    }
  }
}

function initializeShowDetectionLabelControl() {
  const input = document.getElementById('showDetectionLabel');
  if (!input) return;

  try {
    const stored = localStorage.getItem(SHOW_DETECTION_LABEL_STORAGE_KEY);
    if (stored !== null) {
      showDetectionLabel = stored !== '0';
    }
  } catch (_err) { /* ignore */ }

  input.checked = showDetectionLabel;
  input.addEventListener('change', () => {
    showDetectionLabel = input.checked;
    try {
      localStorage.setItem(SHOW_DETECTION_LABEL_STORAGE_KEY, showDetectionLabel ? '1' : '0');
    } catch (_err) { /* ignore */ }
    drawOverlay();
  });
}

function initializeUpdateIntervalControl() {
  const input = document.getElementById('updateIntervalInput');
  if (!input) return;

  try {
    const stored = localStorage.getItem(UPDATE_INTERVAL_STORAGE_KEY);
    if (stored !== null) {
      const parsed = parseInt(stored, 10);
      if (parsed >= 100) updateIntervalMs = parsed;
    }
  } catch (_err) { /* ignore */ }

  input.value = String(updateIntervalMs);
  input.addEventListener('change', () => {
    const val = Math.max(100, parseInt(input.value, 10) || updateIntervalMs);
    input.value = String(val);
    updateIntervalMs = val;
    try {
      localStorage.setItem(UPDATE_INTERVAL_STORAGE_KEY, String(val));
    } catch (_err) { /* ignore */ }
  });
}

function initializeMirrorViewControl() {
  const mirrorViewInput = document.getElementById('mirrorView');
  if (!mirrorViewInput) {
    return;
  }

  let initialMirrorState = false;
  try {
    initialMirrorState = localStorage.getItem(MIRROR_VIEW_STORAGE_KEY) === '1';
  } catch (_err) {
    initialMirrorState = false;
  }

  setMirrorEnabled(initialMirrorState, { persist: false });
  mirrorViewInput.addEventListener('change', () => {
    setMirrorEnabled(mirrorViewInput.checked);
  });
}

function setStreamActive(active) {
  const shouldBeActive = !!active;
  if (streamActive === shouldBeActive) {
    return;
  }

  streamActive = shouldBeActive;
  if (streamActive) {
    video.src = STREAM_FEED_URL;
    scheduleResizeSync(10);
    return;
  }

  // Switching to a local data URI closes the active MJPEG network stream.
  video.src = 'data:,';
}

function getDefaultPanelVisibility() {
  return {
    backendView: true,
    streamView: true,
    voteInfoView: true,
    countView: true,
  };
}

function loadPanelVisibility() {
  const defaults = getDefaultPanelVisibility();
  try {
    const raw = localStorage.getItem(PANEL_VISIBILITY_STORAGE_KEY);
    if (!raw) {
      return defaults;
    }
    const parsed = JSON.parse(raw);
    return {
      backendView: isPanelVisible(parsed.backendView),
      streamView: isPanelVisible(parsed.streamView),
      voteInfoView: isPanelVisible(parsed.voteInfoView),
      countView: isPanelVisible(parsed.countView),
    };
  } catch (_err) {
    return defaults;
  }
}

function persistPanelVisibility(visibility) {
  try {
    localStorage.setItem(PANEL_VISIBILITY_STORAGE_KEY, JSON.stringify(visibility));
  } catch (_err) {
    // Keep runtime behavior even if persistence is unavailable.
  }
}

function isPanelVisible(value) {
  if (value === false || value === 0) return false;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'false' || normalized === '0' || normalized === 'off' || normalized === 'no') {
      return false;
    }
  }
  return true;
}

function setPanelVisible(panelElement, visible) {
  if (!panelElement) return;
  panelElement.classList.toggle('panel-hidden', !visible);
  // Inline fallback prevents accidental CSS regressions from keeping a panel visible.
  panelElement.style.display = visible ? '' : 'none';
}

function applyPanelVisibility(visibility, { persist = true } = {}) {
  const normalized = {
    backendView: isPanelVisible(visibility.backendView),
    streamView: isPanelVisible(visibility.streamView),
    voteInfoView: isPanelVisible(visibility.voteInfoView),
    countView: isPanelVisible(visibility.countView),
  };

  setPanelVisible(backendView, normalized.backendView);
  setPanelVisible(streamView, normalized.streamView);
  setPanelVisible(voteInfoView, normalized.voteInfoView);
  setPanelVisible(countView, normalized.countView);
  page.classList.toggle('backend-hidden', !normalized.backendView);

  if (toggleBackendViewInput) toggleBackendViewInput.checked = normalized.backendView;
  if (toggleStreamViewInput) toggleStreamViewInput.checked = normalized.streamView;
  if (toggleVoteInfoViewInput) toggleVoteInfoViewInput.checked = normalized.voteInfoView;
  if (toggleCountViewInput) toggleCountViewInput.checked = normalized.countView;

  setStreamActive(normalized.streamView);
  scheduleResizeSync(8);

  if (persist) {
    persistPanelVisibility(normalized);
  }
}

function collectPanelVisibilityFromInputs() {
  return {
    backendView: toggleBackendViewInput ? toggleBackendViewInput.checked : true,
    streamView: toggleStreamViewInput ? toggleStreamViewInput.checked : true,
    voteInfoView: toggleVoteInfoViewInput ? toggleVoteInfoViewInput.checked : true,
    countView: toggleCountViewInput ? toggleCountViewInput.checked : true,
  };
}

function initializePanelVisibilityControls() {
  const initialVisibility = loadPanelVisibility();
  applyPanelVisibility(initialVisibility, { persist: false });

  if (panelToggleContent) {
    panelToggleContent.addEventListener('change', () => {
      applyPanelVisibility(collectPanelVisibilityFromInputs());
    });
  }

  if (toggleBackendViewInput) {
    toggleBackendViewInput.addEventListener('change', () => {
      applyPanelVisibility(collectPanelVisibilityFromInputs());
    });
  }

  if (toggleStreamViewInput) {
    toggleStreamViewInput.addEventListener('change', () => {
      applyPanelVisibility(collectPanelVisibilityFromInputs());
    });
  }

  if (toggleVoteInfoViewInput) {
    toggleVoteInfoViewInput.addEventListener('change', () => {
      applyPanelVisibility(collectPanelVisibilityFromInputs());
    });
  }

  if (toggleCountViewInput) {
    toggleCountViewInput.addEventListener('change', () => {
      applyPanelVisibility(collectPanelVisibilityFromInputs());
    });
  }
}

function setPanelToolbarOpen(isOpen) {
  if (!panelToggleBar || !panelToggleHandle) {
    return;
  }
  panelToggleBar.classList.toggle('is-open', !!isOpen);
  panelToggleHandle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}

function syncPanelToolbarOffset() {
  if (!panelToggleBar || !panelToggleContent) {
    return;
  }
  const hiddenOffset = Math.max(0, Math.ceil(panelToggleContent.getBoundingClientRect().height));
  panelToggleBar.style.setProperty('--panel-toggle-hidden-offset', `${hiddenOffset}px`);
}

function initializePanelToolbarAutoHide() {
  if (!panelToggleBar || !panelToggleHandle || !panelToggleContent) {
    return;
  }

  syncPanelToolbarOffset();

  panelToggleHandle.addEventListener('click', (event) => {
    event.stopPropagation();
    setPanelToolbarOpen(!panelToggleBar.classList.contains('is-open'));
  });

  panelToggleBar.addEventListener('mouseleave', () => {
    setPanelToolbarOpen(false);
  });

  document.addEventListener('click', (event) => {
    if (!panelToggleBar.contains(event.target)) {
      setPanelToolbarOpen(false);
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      setPanelToolbarOpen(false);
    }
  });

  panelToggleBar.addEventListener('focusout', () => {
    window.setTimeout(() => {
      if (!panelToggleBar.contains(document.activeElement)) {
        setPanelToolbarOpen(false);
      }
    }, 0);
  });

  window.addEventListener('resize', syncPanelToolbarOffset);

  if (typeof ResizeObserver !== 'undefined') {
    const panelToolbarObserver = new ResizeObserver(() => syncPanelToolbarOffset());
    panelToolbarObserver.observe(panelToggleContent);
  }
}

function clampPointToFrame(x, y) {
  const frameWidth = hasFrameState() ? state.frame.width : getCanvasDisplayWidth();
  const frameHeight = hasFrameState() ? state.frame.height : getCanvasDisplayHeight();
  return [
    Math.max(0, Math.min(frameWidth - 1, Math.round(x))),
    Math.max(0, Math.min(frameHeight - 1, Math.round(y)))
  ];
}

function isPointInPolygon(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = toDisplayPoint(points[i]);
    const [xj, yj] = toDisplayPoint(points[j]);
    const intersect = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function createCenteredRectanglePoints() {
  const frameWidth = hasFrameState() ? state.frame.width : 960;
  const frameHeight = hasFrameState() ? state.frame.height : 540;
  const rectWidth = Math.max(80, Math.round(frameWidth * 0.25));
  const rectHeight = Math.max(60, Math.round(frameHeight * 0.2));
  const centerX = Math.round(frameWidth / 2);
  const centerY = Math.round(frameHeight / 2);
  const left = centerX - Math.round(rectWidth / 2);
  const right = centerX + Math.round(rectWidth / 2);
  const top = centerY - Math.round(rectHeight / 2);
  const bottom = centerY + Math.round(rectHeight / 2);
  return [
    clampPointToFrame(left, top),
    clampPointToFrame(right, top),
    clampPointToFrame(right, bottom),
    clampPointToFrame(left, bottom)
  ];
}

function drawPolygon(points, color, dashed = false, showVertices = false) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color + '22';
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([6, 4]);
  ctx.beginPath();
  points.forEach((p, idx) => {
    const dp = toDisplayPoint(p);
    if (idx === 0) ctx.moveTo(dp[0], dp[1]);
    else ctx.lineTo(dp[0], dp[1]);
  });
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
  if (showVertices) {
    points.forEach((p) => {
      const dp = toDisplayPoint(p);
      ctx.beginPath();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.arc(dp[0], dp[1], 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }
  ctx.restore();
}

function getSelectedRegion() {
  if (selectedRegionId === null) return null;
  return regions.find((r) => Number(r.id) === Number(selectedRegionId)) || null;
}

function setRegions(nextRegions) {
  regions = Array.isArray(nextRegions) ? nextRegions : [];
  if (!regions.some((r) => Number(r.id) === Number(selectedRegionId))) {
    selectedRegionId = regions.length ? Number(regions[0].id) : null;
  }
}

function markRegionsDirty() {
  regionsDirty = true;
}

function getCurrentQuestionAnswers() {
  if (!state || !state.voting || !state.voting.current_question) {
    return [];
  }
  return state.voting.current_question.answers || [];
}

function getAnswerSlotName(slot, { preferQuestionAnswer = false } = {}) {
  const normalizedSlot = Number(slot) === 2 ? 2 : 1;
  const answers = getCurrentQuestionAnswers();
  const answerText = answers[normalizedSlot - 1];
  if (preferQuestionAnswer && answerText) {
    return answerText;
  }
  if (answerText) {
    return `Answer ${normalizedSlot} (${answerText})`;
  }
  return `Answer ${normalizedSlot}`;
}

function syncRegionEditorFromSelected() {
  const region = getSelectedRegion();
  document.getElementById('regionAnswerSlot').value = region ? String(region.answer_slot || 1) : '1';
  const color1Input = document.getElementById('answer1Color');
  const color2Input = document.getElementById('answer2Color');
  if (document.activeElement !== color1Input) color1Input.value = getSlotColor(1);
  if (document.activeElement !== color2Input) color2Input.value = getSlotColor(2);
  if (!region) return;
  document.getElementById('regionCriterion').value = region.criterion || 'overlap';
}

function drawDetections() {
  if (!hasFrameState()) return;
  const displayWidth = getCanvasDisplayWidth();
  const displayHeight = getCanvasDisplayHeight();
  const fontPx = Math.max(12, Math.round(Math.min(displayWidth, displayHeight) * 0.018));

  (state.assignments || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.box;
    const [dx1, dy1] = toDisplayPoint([x1, y1]);
    const [dx2, dy2] = toDisplayPoint([x2, y2]);
    const dx = Math.min(dx1, dx2);
    const dy = Math.min(dy1, dy2);
    const dw = Math.abs(dx2 - dx1);
    const dh = Math.abs(dy2 - dy1);
    const color = det.region_answer_slot ? getSlotColor(det.region_answer_slot) : '#b9c0ce';
    const assignedText = det.region_display_name || 'unassigned';
    const caption = showDetectionLabel
      ? `${det.label} ${Number(det.score || 0).toFixed(2)} -> ${assignedText}`
      : assignedText;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.5, Math.min(3, displayWidth / 640));
    ctx.strokeRect(dx, dy, dw, dh);

    ctx.font = `600 ${fontPx}px "Space Grotesk", sans-serif`;
    const textPaddingX = 8;
    const textPaddingY = 4;
    const textMetrics = ctx.measureText(caption);
    const textWidth = Math.ceil(textMetrics.width);
    const boxHeight = fontPx + textPaddingY * 2;
    const boxY = Math.max(0, dy - boxHeight - 4);
    const boxX = Math.max(0, dx);

    ctx.fillStyle = color;
    ctx.fillRect(boxX, boxY, textWidth + textPaddingX * 2, boxHeight);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(caption, boxX + textPaddingX, boxY + fontPx + textPaddingY - 2);
    ctx.restore();
  });
}

function drawOverlay() {
  ctx.clearRect(0, 0, getCanvasDisplayWidth(), getCanvasDisplayHeight());
  drawDetections();
  regions.forEach((r, index) => {
    const isSelected = Number(r.id) === Number(selectedRegionId);
    drawPolygon(r.points, getSlotColor(r.answer_slot || ((index % 2) + 1)), false, isSelected);
  });
}

function refreshRegionList() {
  const host = document.getElementById('regionList');
  if (!regions.length) {
    host.innerHTML = 'No regions yet.';
    selectedRegionId = null;
    syncRegionEditorFromSelected();
    return;
  }
  if (!regions.some((r) => Number(r.id) === Number(selectedRegionId))) {
    selectedRegionId = Number(regions[0].id);
  }
  host.innerHTML = regions.map((r) => {
    const cls = Number(r.id) === Number(selectedRegionId) ? 'region-item selected' : 'region-item';
    const slotName = getAnswerSlotName(r.answer_slot || 1);
    return `<div class="${cls}" data-id="${r.id}">#${r.id} <b>${slotName}</b> (${r.criterion}) - ${r.points.length} pts</div>`;
  }).join('');
  Array.from(host.querySelectorAll('.region-item')).forEach((el) => {
    el.addEventListener('click', () => {
      selectedRegionId = Number(el.dataset.id);
      syncRegionEditorFromSelected();
      refreshRegionList();
      drawOverlay();
    });
  });
  syncRegionEditorFromSelected();
}

async function saveRegions() {
  if (regionSaveInFlight) {
    return false;
  }

  regionSaveInFlight = true;
  try {
    const resp = await fetch('/set_regions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ regions })
    });

    if (!resp.ok) {
      throw new Error(`Region save failed with HTTP ${resp.status}`);
    }

    const data = await resp.json();
    if (!data || data.status !== 'success') {
      throw new Error((data && data.message) ? data.message : 'Region save failed');
    }

    setRegions(data.regions || []);
    regionsDirty = false;
    refreshRegionList();
    drawOverlay();
    flash(data.message || 'Regions saved');
    return true;
  } catch (error) {
    regionsDirty = true;
    flash(`Failed to save regions: ${error.message}`);
    return false;
  } finally {
    regionSaveInFlight = false;
  }
}

async function persistRegions(message) {
  const saved = await saveRegions();
  if (saved && message) {
    flash(message);
  }
  return saved;
}

async function finishPointDrag() {
  const didMove = pointDragMoved;
  draggingPointIdx = null;
  draggingRegion = false;
  dragLastX = null;
  dragLastY = null;
  pointDragMoved = false;
  if (didMove) {
    await persistRegions('Region shape saved.');
  }
}

function rebuildChart(history) {
  if (!history || !history.length) {
    chart.data.labels = [];
    chart.data.datasets = [];
    chart.update();
    return;
  }

  const allLabels = new Set();
  history.forEach((row) => Object.keys(row.counts || {}).forEach((k) => allLabels.add(k)));
  const labels = Array.from(allLabels).sort();

  chart.data.labels = history.map((_, idx) => idx);
  chart.data.datasets = labels.map((label, idx) => ({
    label: getAnswerSlotName(Number(label)),
    data: history.map((row) => (row.counts && row.counts[label]) || 0),
    borderColor: getSlotColor(Number(label)),
    backgroundColor: withAlpha(getSlotColor(Number(label))),
    fill: false,
    tension: 0.15,
  }));
  chart.update();
}

function refreshCounts(labelCounts) {
  const host = document.getElementById('countChips');
  const slotCounts = labelCounts || {};
  const entries = [1, 2].map((slot) => [getAnswerSlotName(slot, { preferQuestionAnswer: true }), Number(slotCounts[String(slot)] || slotCounts[slot] || 0)]);
  if (!entries.length) {
    host.innerHTML = '<div class="chip">No labeled counts</div>';
    return;
  }
  host.innerHTML = entries.map(([k, v], index) => `<div class="chip" style="border-color: ${getSlotColor(index + 1)};"><b>${k}</b>: ${v}</div>`).join('');
}

async function saveSlotColors() {
  const payload = {
    slot_colors: {
      1: document.getElementById('answer1Color').value,
      2: document.getElementById('answer2Color').value,
    }
  };
  const resp = await fetch('/set_slot_colors', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!resp.ok) {
    throw new Error(`Color update failed with HTTP ${resp.status}`);
  }

  const data = await resp.json();
  if (!data || data.status !== 'success') {
    throw new Error((data && data.message) ? data.message : 'Color update failed');
  }

  slotColors = {
    1: data.slot_colors?.['1'] || slotColors[1],
    2: data.slot_colors?.['2'] || slotColors[2],
  };
  drawOverlay();
  rebuildChart(state?.history || []);
  refreshCounts(state?.slot_counts || {});
  flash(data.message || 'Answer colors updated');
}

function updateVotingPhaseDisplay(votingData, slotCounts) {
  const votingPhaseDisplay = document.getElementById('votingPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');

  // Hide other displays
  pausePhaseDisplay.style.display = 'none';
  idlePhaseDisplay.style.display = 'none';
  votingPhaseDisplay.style.display = 'block';

  // Update question
  const question = votingData.current_question;
  if (question) {
    setMarkdownContent('votingQuestion', question.question, '');

    // Update answer blocks
    const answers = question.answers || [];
    const count1 = Number(slotCounts?.[1] || slotCounts?.['1'] || 0);
    const count2 = Number(slotCounts?.[2] || slotCounts?.['2'] || 0);

    // Update Answer 1
    setMarkdownContent('answerText1', answers[0] || '', 'Answer 1');
    document.getElementById('answerCount1').textContent = count1;

    // Update Answer 2
    setMarkdownContent('answerText2', answers[1] || '', 'Answer 2');
    document.getElementById('answerCount2').textContent = count2;
    
    // Update block styling based on colors
    document.getElementById('answerBlock1').style.borderColor = getSlotColor(1);
    document.getElementById('answerBlock2').style.borderColor = getSlotColor(2);
    document.getElementById('answerBlock1').style.background = withAlpha(getSlotColor(1), '10');
    document.getElementById('answerBlock2').style.background = withAlpha(getSlotColor(2), '10');
    
    // Highlight the leading answer
    const block1 = document.getElementById('answerBlock1');
    const block2 = document.getElementById('answerBlock2');
    
    if (count1 > count2) {
      block1.classList.add('leading');
      block2.classList.remove('leading');
      block2.classList.add('minority');
      block1.classList.remove('minority');
    } else if (count2 > count1) {
      block2.classList.add('leading');
      block1.classList.remove('leading');
      block1.classList.add('minority');
      block2.classList.remove('minority');
    } else {
      block1.classList.remove('leading');
      block2.classList.remove('leading');
      block1.classList.remove('minority');
      block2.classList.remove('minority');
    }
    
    // Update timer
    const timeLeft = votingData.time_left_sec || 0;
    document.getElementById('votingTimer').textContent = `Time left: ${timeLeft}s`;
  }
}

function updatePausePhaseDisplay(votingData, slotCounts) {
  const votingPhaseDisplay = document.getElementById('votingPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');

  // Hide other displays
  votingPhaseDisplay.style.display = 'none';
  idlePhaseDisplay.style.display = 'none';
  pausePhaseDisplay.style.display = 'block';

  const lastResult = votingData.last_vote_result;
  if (lastResult) {
    const question = lastResult.question;

    // Update question
    setMarkdownContent('pauseQuestionText', question.question, '');

    // Update counts
    const answers = question.answers || [];
    const count1 = lastResult.counts?.[answers[0]] || 0;
    const count2 = lastResult.counts?.[answers[1]] || 0;

    setMarkdownContent('pauseAnswer1Label', answers[0] || '', 'Answer 1');
    document.getElementById('pauseAnswer1Count').textContent = count1;
    setMarkdownContent('pauseAnswer2Label', answers[1] || '', 'Answer 2');
    document.getElementById('pauseAnswer2Count').textContent = count2;
    
    // Update colors
    document.getElementById('pauseAnswer1').style.borderColor = getSlotColor(1);
    document.getElementById('pauseAnswer2').style.borderColor = getSlotColor(2);
    
    // Remove previous marking classes
    document.getElementById('pauseAnswer1').classList.remove('answer-correct', 'answer-incorrect');
    document.getElementById('pauseAnswer2').classList.remove('answer-correct', 'answer-incorrect');
    
    // Update result message
    const resultMsg = document.getElementById('pauseResultMessage');
    
    if (lastResult.voted_answer === null) {
      resultMsg.textContent = '⚖️ It\'s a tie!';
      resultMsg.className = 'pause-result-message tie';
    } else if (lastResult.is_correct === true) {
      resultMsg.textContent = '✓ Correct!';
      resultMsg.className = 'pause-result-message correct';
    } else if (lastResult.is_correct === false) {
      resultMsg.textContent = '✗ Wrong';
      resultMsg.className = 'pause-result-message incorrect';
    } else {
      resultMsg.textContent = 'Result unknown';
      resultMsg.className = 'pause-result-message tie';
    }
    
    // Apply check/cross marks to answer items
    const correctAnswer = question.correct_answer || 'Not specified';
    const answerBlock1 = document.getElementById('pauseAnswer1');
    const answerBlock2 = document.getElementById('pauseAnswer2');
    
    if (answers[0] === correctAnswer) {
      answerBlock1.classList.add('answer-correct');
      answerBlock2.classList.add('answer-incorrect');
    } else if (answers[1] === correctAnswer) {
      answerBlock1.classList.add('answer-incorrect');
      answerBlock2.classList.add('answer-correct');
    }
    
    // Show correct/wrong answer
    const correctWrongDiv = document.getElementById('pauseCorrectWrong');
    const moreInfo = question.more_info ? `<div class="pause-more-info">${renderBasicMarkdown(question.more_info)}</div>` : '';
    correctWrongDiv.innerHTML = `Correct answer: ${renderBasicMarkdown(correctAnswer) || escapeHtml('Not specified')}${moreInfo}`;
    
    // Update accuracy display
    const accuracyContainer = document.getElementById('pauseAccuracyContainer');
    const recentVotes = votingData.recent_scored_votes || 0;
    if (recentVotes > 0 && votingData.accuracy_percent !== null) {
      accuracyContainer.style.display = 'block';
      const accuracy = votingData.accuracy_percent;
      document.getElementById('pauseAccuracyLabel').textContent = `Recent accuracy (last ${recentVotes} questions):`;
      document.getElementById('pauseAccuracyValue').textContent = `${accuracy}%`;
    } else {
      accuracyContainer.style.display = 'none';
    }
    
    // Update countdown
    const timeLeft = votingData.time_left_sec || 0;
    document.getElementById('pauseCountdown').textContent = `Next question in ${timeLeft}s`;
  }
}

function updateVotingDisplay(votingData, slotCounts) {
  const votingPhaseDisplay = document.getElementById('votingPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');

  if (!votingData.active) {
    // Show idle display
    votingPhaseDisplay.style.display = 'none';
    pausePhaseDisplay.style.display = 'none';
    idlePhaseDisplay.style.display = 'block';
    return;
  }

  if (votingData.phase === 'question') {
    updateVotingPhaseDisplay(votingData, slotCounts);
  } else if (votingData.phase === 'pause') {
    updatePausePhaseDisplay(votingData, slotCounts);
  } else {
    // Fallback to idle
    votingPhaseDisplay.style.display = 'none';
    pausePhaseDisplay.style.display = 'none';
    idlePhaseDisplay.style.display = 'block';
  }
}

function refreshVoting(v) {
  document.getElementById('modeLabel').textContent = 'Mode: ' + (v.active ? 'voting' : 'counting');
  const countChipsHost = document.getElementById('countChips');
  countChipsHost.style.display = v.active ? 'none' : 'flex';

  const voteDurationInput = document.getElementById('voteDuration');
  const pauseDurationInput = document.getElementById('pauseDuration');
  const rollingWindowInput = document.getElementById('rollingWindow');
  if (document.activeElement !== voteDurationInput) voteDurationInput.value = String(v.vote_duration_sec ?? voteDurationInput.value);
  if (document.activeElement !== pauseDurationInput) pauseDurationInput.value = String(v.pause_duration_sec ?? pauseDurationInput.value);
  if (document.activeElement !== rollingWindowInput) rollingWindowInput.value = String(v.window_size ?? rollingWindowInput.value);
  lastVotingConfigSignature = `${Number(v.vote_duration_sec || 0)}|${Number(v.pause_duration_sec || 0)}|${Number(v.window_size || 0)}`;

  // Update voting display
  const slotCounts = state?.slot_counts || {};
  updateVotingDisplay(v, slotCounts);

  // Keep the stats for idle phase
  const statHost = document.getElementById('voteStats');
  const acc = v.accuracy_percent === null ? 'n/a' : `${v.accuracy_percent}%`;
  statHost.innerHTML = [
    `<div class="chip">Accuracy: ${acc}</div>`,
    `<div class="chip">Scored votes: ${v.recent_scored_votes}</div>`,
  ].join('');
}

function readDetectorPayloadFromInputs() {
  const detector = document.getElementById('detectorType').value;
  const objects = document.getElementById('objectsInput').value
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean);
  const threshold = Number(document.getElementById('thresholdInput').value);
  const frameSkip = Number(document.getElementById('frameSkipInput').value);

  if (!detector || !Number.isFinite(threshold) || !Number.isFinite(frameSkip) || !objects.length) {
    return null;
  }

  const frameSkipInt = Math.trunc(frameSkip);
  if (frameSkipInt < 1) {
    return null;
  }

  return { detector, objects, threshold, frame_skip: frameSkipInt };
}

function getDetectorSignature(payload) {
  return `${payload.detector}|${payload.threshold}|${payload.frame_skip}|${payload.objects.join('\\n')}`;
}

function syncDetectorInputsFromState(nextState) {
  if (!nextState || !nextState.detector) return;
  const detectorTypeInput = document.getElementById('detectorType');
  const objectsInput = document.getElementById('objectsInput');
  const thresholdInput = document.getElementById('thresholdInput');
  const frameSkipInput = document.getElementById('frameSkipInput');

  if (document.activeElement !== detectorTypeInput) {
    detectorTypeInput.value = nextState.detector.type || detectorTypeInput.value;
  }
  if (document.activeElement !== objectsInput) {
    objectsInput.value = (nextState.detector.objects || []).join('\n');
  }
  if (document.activeElement !== thresholdInput) {
    thresholdInput.value = String(nextState.detector.threshold ?? thresholdInput.value);
  }
  if (document.activeElement !== frameSkipInput) {
    frameSkipInput.value = String(nextState.detector.frame_skip ?? frameSkipInput.value);
  }

  const payload = {
    detector: nextState.detector.type || detectorTypeInput.value,
    objects: Array.isArray(nextState.detector.objects) ? nextState.detector.objects : [],
    threshold: Number(nextState.detector.threshold ?? thresholdInput.value),
    frame_skip: Math.trunc(Number(nextState.detector.frame_skip ?? frameSkipInput.value)),
  };
  if (payload.detector && Number.isFinite(payload.threshold) && Number.isFinite(payload.frame_skip) && payload.frame_skip >= 1 && payload.objects.length) {
    lastDetectorSignature = getDetectorSignature(payload);
  }
}

async function saveDetectorFromInputs() {
  const payload = readDetectorPayloadFromInputs();
  if (!payload) {
    return;
  }

  const signature = getDetectorSignature(payload);
  if (signature === lastDetectorSignature) {
    return;
  }

  if (detectorSaveInFlight) {
    detectorSaveQueued = true;
    return;
  }

  detectorSaveInFlight = true;
  try {
    const resp = await fetch('/apply_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (!resp.ok || !data || data.status !== 'success') {
      throw new Error((data && data.message) ? data.message : `Detector settings update failed with HTTP ${resp.status}`);
    }
    lastDetectorSignature = signature;
    flash(data.message || 'Detector settings applied');
  } catch (error) {
    flash(`Detector settings update failed: ${error.message}`);
  } finally {
    detectorSaveInFlight = false;
    if (detectorSaveQueued) {
      detectorSaveQueued = false;
      await saveDetectorFromInputs();
    }
  }
}

function queueDetectorSave() {
  if (detectorDebounceHandle !== null) {
    clearTimeout(detectorDebounceHandle);
  }
  detectorDebounceHandle = setTimeout(async () => {
    detectorDebounceHandle = null;
    await saveDetectorFromInputs();
  }, 350);
}

function readVotingConfigPayloadFromInputs() {
  const voteDuration = Number(document.getElementById('voteDuration').value);
  const pauseDuration = Number(document.getElementById('pauseDuration').value);
  const windowSize = Number(document.getElementById('rollingWindow').value);

  if (!Number.isFinite(voteDuration) || !Number.isFinite(pauseDuration) || !Number.isFinite(windowSize)) {
    return null;
  }

  const payload = {
    vote_duration_sec: Math.trunc(voteDuration),
    pause_duration_sec: Math.trunc(pauseDuration),
    window_size: Math.trunc(windowSize)
  };

  if (payload.vote_duration_sec < 1 || payload.pause_duration_sec < 0 || payload.window_size < 1) {
    return null;
  }

  return payload;
}

function getVotingConfigSignature(payload) {
  return `${payload.vote_duration_sec}|${payload.pause_duration_sec}|${payload.window_size}`;
}

async function saveVotingConfigFromInputs() {
  const payload = readVotingConfigPayloadFromInputs();
  if (!payload) {
    return;
  }

  const signature = getVotingConfigSignature(payload);
  if (signature === lastVotingConfigSignature) {
    return;
  }

  if (votingConfigSaveInFlight) {
    votingConfigSaveQueued = true;
    return;
  }

  votingConfigSaveInFlight = true;
  try {
    const resp = await fetch('/voting_config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (!resp.ok || !data || data.status !== 'success') {
      throw new Error((data && data.message) ? data.message : `Voting config update failed with HTTP ${resp.status}`);
    }
    lastVotingConfigSignature = signature;
    flash(data.message || 'Voting config saved');
  } catch (error) {
    flash(`Voting config update failed: ${error.message}`);
  } finally {
    votingConfigSaveInFlight = false;
    if (votingConfigSaveQueued) {
      votingConfigSaveQueued = false;
      await saveVotingConfigFromInputs();
    }
  }
}

function queueVotingConfigSave() {
  if (votingConfigDebounceHandle !== null) {
    clearTimeout(votingConfigDebounceHandle);
  }
  votingConfigDebounceHandle = setTimeout(async () => {
    votingConfigDebounceHandle = null;
    await saveVotingConfigFromInputs();
  }, 300);
}

async function fetchState() {
  const resp = await fetch('/state_json');
  if (!resp.ok) {
    flash('State endpoint HTTP error: ' + resp.status);
    return;
  }

  const nextState = await resp.json();
  if (!nextState || nextState.status !== 'success' || !nextState.frame) {
    flash((nextState && nextState.message) ? nextState.message : 'Invalid state payload');
    return;
  }

  state = nextState;
  syncDetectorInputsFromState(nextState);
  slotColors = {
    1: nextState.slot_colors?.['1'] || slotColors[1],
    2: nextState.slot_colors?.['2'] || slotColors[2],
  };
  if (!regionsDirty) {
    setRegions(state.regions || []);
  }
  refreshRegionList();
  refreshCounts(state.slot_counts || {});
  refreshVoting(state.voting || {});
  rebuildChart(state.history || []);
  drawOverlay();
}

canvas.addEventListener('pointerdown', (ev) => {
  const region = getSelectedRegion();
  if (!region) return;
  const { x, y } = getCanvasCoordinates(ev);
  let bestIdx = null;
  let bestDist = Infinity;
  region.points.forEach((p, idx) => {
    const dp = toDisplayPoint(p);
    const dx = dp[0] - x;
    const dy = dp[1] - y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = idx;
    }
  });
  if (bestIdx !== null && bestDist <= 14) {
    ev.preventDefault();
    draggingPointIdx = bestIdx;
    activePointerId = ev.pointerId;
    pointDragMoved = false;
    canvas.setPointerCapture(ev.pointerId);
  }
  else if (isPointInPolygon(x, y, region.points)) {
    ev.preventDefault();
    draggingRegion = true;
    dragLastX = x;
    dragLastY = y;
    activePointerId = ev.pointerId;
    pointDragMoved = false;
    canvas.setPointerCapture(ev.pointerId);
  }
});

canvas.addEventListener('pointermove', (ev) => {
  const region = getSelectedRegion();
  if (!region) return;
  if (activePointerId !== ev.pointerId) return;
  ev.preventDefault();
  const { x, y } = getCanvasCoordinates(ev);
  if (draggingPointIdx !== null) {
    region.points[draggingPointIdx] = toFramePoint(x, y);
    pointDragMoved = true;
    markRegionsDirty();
    drawOverlay();
  } else if (draggingRegion) {
    const scaleX = hasFrameState() ? state.frame.width / getCanvasDisplayWidth() : 1;
    const scaleY = hasFrameState() ? state.frame.height / getCanvasDisplayHeight() : 1;
    const rawDx = Math.round((x - dragLastX) * scaleX);
    const dx = mirrorEnabled ? -rawDx : rawDx;
    const dy = Math.round((y - dragLastY) * scaleY);
    region.points = region.points.map(([px, py]) => clampPointToFrame(px + dx, py + dy));
    dragLastX = x;
    dragLastY = y;
    pointDragMoved = true;
    markRegionsDirty();
    drawOverlay();
  }
});

canvas.addEventListener('pointerup', async (ev) => {
  if (activePointerId === ev.pointerId && canvas.hasPointerCapture(ev.pointerId)) {
    canvas.releasePointerCapture(ev.pointerId);
  }
  activePointerId = null;
  await finishPointDrag();
});

canvas.addEventListener('pointercancel', async (ev) => {
  if (activePointerId === ev.pointerId && canvas.hasPointerCapture(ev.pointerId)) {
    canvas.releasePointerCapture(ev.pointerId);
  }
  activePointerId = null;
  await finishPointDrag();
});

canvas.addEventListener('mouseleave', async () => {
  activePointerId = null;
  await finishPointDrag();
});

document.getElementById('createRegion').onclick = async () => {
  const answerSlot = Number(document.getElementById('regionAnswerSlot').value || '1');
  const criterion = document.getElementById('regionCriterion').value;
  const nextId = regions.length ? Math.max(...regions.map((r) => r.id || 0)) + 1 : 1;
  regions.push({ id: nextId, answer_slot: answerSlot, criterion, points: createCenteredRectanglePoints() });
  selectedRegionId = nextId;
  markRegionsDirty();
  refreshRegionList();
  drawOverlay();
  await persistRegions('Region created.');
};

async function updateSelectedRegionFromControls(message) {
  const region = getSelectedRegion();
  if (!region) {
    syncRegionEditorFromSelected();
    return;
  }
  const answerSlot = Number(document.getElementById('regionAnswerSlot').value || '1');
  const criterion = document.getElementById('regionCriterion').value;
  region.answer_slot = answerSlot;
  region.criterion = criterion;
  markRegionsDirty();
  refreshRegionList();
  drawOverlay();
  await persistRegions(message);
}

document.getElementById('regionAnswerSlot').addEventListener('change', async () => {
  await updateSelectedRegionFromControls('Selected region updated.');
});

document.getElementById('regionCriterion').addEventListener('change', async () => {
  await updateSelectedRegionFromControls('Selected region updated.');
});

document.getElementById('answer1Color').addEventListener('change', async () => {
  await saveSlotColors();
});

document.getElementById('answer2Color').addEventListener('change', async () => {
  await saveSlotColors();
});

document.getElementById('deleteSelectedRegion').onclick = async () => {
  const region = getSelectedRegion();
  if (!region) {
    flash('Select a region first.');
    return;
  }
  regions = regions.filter((r) => Number(r.id) !== Number(region.id));
  selectedRegionId = regions.length ? Number(regions[0].id) : null;
  draggingPointIdx = null;
  activePointerId = null;
  pointDragMoved = false;
  markRegionsDirty();
  syncRegionEditorFromSelected();
  refreshRegionList();
  drawOverlay();
  await persistRegions('Selected region deleted.');
};

document.getElementById('detectorType').addEventListener('change', queueDetectorSave);
document.getElementById('objectsInput').addEventListener('input', queueDetectorSave);
document.getElementById('objectsInput').addEventListener('change', queueDetectorSave);
document.getElementById('thresholdInput').addEventListener('input', queueDetectorSave);
document.getElementById('thresholdInput').addEventListener('change', queueDetectorSave);
document.getElementById('frameSkipInput').addEventListener('input', queueDetectorSave);
document.getElementById('frameSkipInput').addEventListener('change', queueDetectorSave);

['voteDuration', 'pauseDuration', 'rollingWindow'].forEach((id) => {
  const el = document.getElementById(id);
  el.addEventListener('input', queueVotingConfigSave);
  el.addEventListener('change', queueVotingConfigSave);
});

document.getElementById('startVoting').onclick = async () => {
  const resp = await fetch('/voting_start', { method: 'POST' });
  const data = await resp.json();
  flash(data.message || 'Voting started');
};

document.getElementById('stopVoting').onclick = async () => {
  const resp = await fetch('/voting_stop', { method: 'POST' });
  const data = await resp.json();
  flash(data.message || 'Voting stopped');
};

document.getElementById('loadQuestions').onclick = async () => {
  const text = document.getElementById('questionsJson').value.trim();
  if (!text) {
    flash('Paste a JSON array or object with questions first.');
    return;
  }
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    flash('Invalid JSON content.');
    return;
  }
  const resp = await fetch('/voting_questions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  flash(data.message || 'Questions updated');
};

window.addEventListener('resize', () => scheduleResizeSync(12));
window.addEventListener('load', () => scheduleResizeSync(16));
video.addEventListener('load', () => scheduleResizeSync(16));

if (typeof ResizeObserver !== 'undefined') {
  const resizeObserver = new ResizeObserver(() => scheduleResizeSync(6));
  resizeObserver.observe(video);
  resizeObserver.observe(videoWrap);
}

initializeShowDetectionLabelControl();
initializeMirrorViewControl();
initializeUpdateIntervalControl();
initializePanelVisibilityControls();
initializePanelToolbarAutoHide();

async function loop() {
  try {
    await fetchState();
  } catch (e) {
    flash('State fetch failed: ' + e.message);
  }
  setTimeout(loop, updateIntervalMs);
}

loop();
scheduleResizeSync(16);
