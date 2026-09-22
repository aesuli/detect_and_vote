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
let detectorInputsInitialized = false;
let canvasDisplayWidth = 1;
let canvasDisplayHeight = 1;
let resizeSyncFrameHandle = null;
let mirrorEnabled = false;
let showDetectionLabel = true;
let showDetectionAnswer = true;

const MIRROR_VIEW_STORAGE_KEY = 'look_and_detect_mirror_view';
const SHOW_DETECTION_LABEL_STORAGE_KEY = 'look_and_detect_show_detection_label';
const SHOW_DETECTION_ANSWER_STORAGE_KEY = 'look_and_detect_show_detection_answer';
const PANEL_VISIBILITY_STORAGE_KEY = 'look_and_detect_panel_visibility';
const UPDATE_INTERVAL_STORAGE_KEY = 'look_and_detect_update_interval';
const LANGUAGE_STORAGE_KEY = 'look_and_detect_language';
const STREAM_FEED_URL = '/video_feed';
let streamActive = true;
let updateIntervalMs = parseInt(document.currentScript?.dataset.updateInterval, 10) || 650;
const markdownContentCache = new WeakMap();
let currentQuestionSource = null;
let questionSources = [];
let questionLoadInFlight = false;
let questionLoadQueued = false;
let countdownTipKey = null;
let previousCountdownTipKey = null;

const COUNTDOWN_TIP_KEYS = [
  'voting.tip.countAtEnd',
  'voting.tip.videoPrivacy',
];

const translations = window.DETECT_AND_VOTE_TRANSLATIONS || {};

let currentLanguage = 'en';

function t(key, values = {}) {
  const template = translations[currentLanguage]?.[key] || translations.en[key] || key;
  return template.replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? `{${name}}`));
}

function applyTranslations() {
  document.documentElement.lang = currentLanguage;
  document.querySelectorAll('[data-i18n]').forEach((element) => {
    const key = element.dataset.i18n;
    const value = element.dataset.i18nValue;
    element.textContent = value === undefined ? t(key) : t(key, { value });
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((element) => {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  });
  const languageSelect = document.getElementById('languageSelect');
  if (languageSelect) languageSelect.value = currentLanguage;
  document.title = t('page.title');
  updateVotingModeUi();
  refreshRegionList();
  refreshCounts(state?.slot_counts || {});
  if (state?.voting) refreshVoting(state.voting);
}

function initializeLanguageControl() {
  try {
    const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (stored && translations[stored]) currentLanguage = stored;
  } catch (_err) { /* ignore */ }
  const languageSelect = document.getElementById('languageSelect');
  if (languageSelect) {
    Object.keys(translations).sort().forEach((languageCode) => {
      const option = document.createElement('option');
      option.value = languageCode;
      option.textContent = translations[languageCode]['language.name'] || languageCode;
      languageSelect.appendChild(option);
    });
  }
  languageSelect?.addEventListener('change', () => {
    currentLanguage = translations[languageSelect.value] ? languageSelect.value : 'en';
    try { localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage); } catch (_err) { /* ignore */ }
    applyTranslations();
  });
  applyTranslations();
}

const chartCanvas = document.getElementById('countChart');
let chart = null;
if (typeof Chart !== 'undefined' && chartCanvas) {
  try {
    chart = new Chart(chartCanvas.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        animation: false,
        responsive: true,
        plugins: { legend: { position: 'bottom' } },
        scales: { x: { display: false }, y: { beginAtZero: true } }
      }
    });
  } catch (error) {
    console.warn('Count chart disabled:', error);
  }
}

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
  if (!urlStr) return false;

  // Check if URL ends with image file extensions (jpg, jpeg, png, webm, gif, svg, webp, bmp).
  // Works for absolute URLs, relative paths, and local app routes.
  const pathOnly = urlStr.split(/[?#]/)[0];
  return /\.(jpg|jpeg|png|webm|gif|svg|webp|bmp)$/i.test(pathOnly);
}

function renderInlineMarkdown(line) {
  let html = escapeHtml(line);
  html = html.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  return html;
}

function buildQuestionAssetUrl(relativePath) {
  if (!currentQuestionSource) {
    return relativePath;
  }

  const normalized = String(relativePath || '')
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\/+/, '')
    .replace(/\/\/+/, '/');

  if (!normalized || normalized === '.' || normalized.startsWith('../')) {
    return relativePath;
  }

  const segments = normalized.split('/').filter(Boolean).map((part) => encodeURIComponent(part));
  if (!segments.length) {
    return relativePath;
  }

  return `/question_asset/${encodeURIComponent(currentQuestionSource)}/${segments.join('/')}`;
}

function resolveQuestionUrl(url) {
  const value = String(url || '').trim();
  if (!value) {
    return value;
  }

  const lower = value.toLowerCase();
  if (lower.startsWith('http://') || lower.startsWith('https://') || lower.startsWith('data:') || lower.startsWith('blob:') || lower.startsWith('mailto:') || lower.startsWith('tel:') || lower.startsWith('javascript:')) {
    return value;
  }

  if (value.startsWith('/') || value.startsWith('#')) {
    return value;
  }

  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value)) {
    return value;
  }

  return buildQuestionAssetUrl(value);
}

function renderBasicMarkdown(markdownText) {
  const normalized = String(markdownText ?? '').trim();
  if (!normalized) {
    return '';
  }

  let html = normalized;

  html = html.replace(/(<(?:img|a)\b[^>]*?\b(?:src|href)\s*=\s*["'])([^"']+)(["'])/gi, (match, prefix, target, suffix) => {
    return `${prefix}${resolveQuestionUrl(target)}${suffix}`;
  });

  // Preserve raw HTML img tags by temporarily replacing them
  const imgTags = [];
  html = html.replace(/<img\s+[^>]*>/gi, (match) => {
    imgTags.push(match);
    return `__IMG_PLACEHOLDER_${imgTags.length - 1}__`;
  });

  const blocks = html.split(/\n\s*\n+/);
  const renderedBlocks = blocks.map((block) => {
    let trimmed = block.trim();
    // Protect markdown image syntax so generic URL replacement cannot corrupt it.
    const markdownImageTokens = [];
    trimmed = trimmed.replace(/!\[([^\]]*)\]\(([^\s)]+)\)/g, (match, alt, url) => {
      const token = `__MD_IMG_TOKEN_${markdownImageTokens.length}__`;
      markdownImageTokens.push({ alt: String(alt || ''), url: String(url || '') });
      return token;
    });


    // Check if this is only a placeholder - restore it as-is
    if (/^__IMG_PLACEHOLDER_\d+__$/.test(trimmed)) {
      const match = trimmed.match(/__IMG_PLACEHOLDER_(\d+)__/);
      if (match) {
        return imgTags[parseInt(match[1])];
      }
    }

    // Check for markdown image syntax ![alt](url)
    const markdownImage = trimmed.match(/^!\[([^\]]*)\]\(([^\s)]+)\)$/i);
    if (markdownImage && isImageUrl(markdownImage[2])) {
      const alt = escapeHtml(markdownImage[1] || 'Question image');
      const src = resolveQuestionUrl(markdownImage[2]);
      return `<img src="${src}" alt="${alt}" loading="lazy" />`;
    }

    // Check if entire block is a plain image URL
    if (isImageUrl(trimmed)) {
      return `<img src="${resolveQuestionUrl(trimmed)}" alt="Question image" loading="lazy" />`;
    }

    // Replace image URLs within text with img tags, preserving surrounding text.
    // This handles URLs and relative image paths.
    trimmed = trimmed.replace(/((?:https?:\/\/|\/)?[^\s<>"{}|\\^`\[\]]*\.(?:jpg|jpeg|png|webm|gif|svg|webp|bmp)(?:\?[^\s<>"{}|\\^`\[\]]*)?)/gi,
      (url) => `<img src="${resolveQuestionUrl(url)}" alt="Image" loading="lazy" />`
    );

    // Restore protected markdown image tokens back to markdown syntax for line-level handling below.
    trimmed = trimmed.replace(/__MD_IMG_TOKEN_(\d+)__/g, (full, idxText) => {
      const idx = Number(idxText);
      const payload = markdownImageTokens[idx];
      if (!payload) {
        return full;
      }
      return `![${payload.alt}](${payload.url})`;
    });

    // Render block line-by-line so markdown image syntax works even when a block
    // contains multiple image lines (for example A/B option images).
    const renderMarkdownImageSegment = (segment) => {
      const imageMatch = segment.match(/^!\[([^\]]*)\]\(([^\s)]+)\)$/i);
      if (!imageMatch) {
        return null;
      }
      if (!isImageUrl(imageMatch[2])) {
        return null;
      }
      const alt = escapeHtml(imageMatch[1] || 'Question image');
      const src = resolveQuestionUrl(imageMatch[2]);
      return `<img src="${src}" alt="${alt}" loading="lazy" />`;
    };

    const renderTextLine = (line) => {
      const segments = line.split(/(<img\b[^>]*>|__IMG_PLACEHOLDER_\d+__|!\[[^\]]*\]\([^\s)]+\))/g);
      return segments
        .map((segment) => {
          if (/^(<img\b[^>]*>|__IMG_PLACEHOLDER_\d+__)$/.test(segment)) {
            return segment;
          }
          const renderedMarkdownImage = renderMarkdownImageSegment(segment);
          if (renderedMarkdownImage !== null) {
            return renderedMarkdownImage;
          }
          return renderInlineMarkdown(segment);
        })
        .join('');
    };

    let result = trimmed
      .split('\n')
      .map((line) => {
        const lineTrimmed = line.trim();
        const lineMarkdownImage = lineTrimmed.match(/^!\[([^\]]*)\]\(([^\s)]+)\)$/i);
        if (lineMarkdownImage && isImageUrl(lineMarkdownImage[2])) {
          const alt = escapeHtml(lineMarkdownImage[1] || 'Question image');
          const src = resolveQuestionUrl(lineMarkdownImage[2]);
          return `<img src="${src}" alt="${alt}" loading="lazy" />`;
        }
        if (isImageUrl(lineTrimmed)) {
          const src = resolveQuestionUrl(lineTrimmed);
          return `<img src="${src}" alt="Question image" loading="lazy" />`;
        }
        return renderTextLine(line);
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

function getVotingMode() {
  return document.getElementById('votingMode')?.value || state?.detector?.voting_mode || 'region_slots';
}

function isObjectListVotingMode() {
  return getVotingMode() === 'object_lists';
}

function parseTextareaList(value) {
  return String(value ?? '')
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean);
}

function ensureObjectListDefaultsInInputs() {
  const answer1Input = document.getElementById('answer1ObjectsInput');
  const answer2Input = document.getElementById('answer2ObjectsInput');
  if (!answer1Input || !answer2Input) {
    return;
  }

  const currentAnswer1 = parseTextareaList(answer1Input.value);
  const currentAnswer2 = parseTextareaList(answer2Input.value);
  const stateAnswer1 = parseTextareaList(state?.detector?.answer_objects?.['1'] || []);
  const stateAnswer2 = parseTextareaList(state?.detector?.answer_objects?.['2'] || []);

  const defaultAnswer1 = stateAnswer1.length ? stateAnswer1 : ['the palm of an open hand'];
  const defaultAnswer2 = stateAnswer2.length ? stateAnswer2 : ['a hand closed in a fist'];

  if (!currentAnswer1.length) {
    answer1Input.value = defaultAnswer1.join('\n');
  }
  if (!currentAnswer2.length) {
    answer2Input.value = defaultAnswer2.join('\n');
  }
}

function getObjectListSignature(items) {
  return items.map((item) => item.toLowerCase()).join('\n');
}

function describeAssignment(det) {
  if (det.vote_display_name) return det.vote_display_name;
  if (det.region_display_name) return det.region_display_name;
  if (det.assignment_reason === 'outside_vote_area') return currentLanguage === 'it' ? 'fuori area di voto' : 'outside vote area';
  if (det.assignment_reason === 'label_not_mapped') return currentLanguage === 'it' ? 'nessuna risposta corrispondente' : 'no answer match';
  return currentLanguage === 'it' ? 'non assegnato' : 'unassigned';
}

function getRegionColor(region, index, isSelected = false) {
  if (isObjectListVotingMode()) {
    return isSelected ? '#136f63' : '#6d7a92';
  }
  return getSlotColor(region.answer_slot || ((index % 2) + 1));
}

function updateVotingModeUi(mode = getVotingMode()) {
  const isObjectMode = mode === 'object_lists';
  const votingModeInput = document.getElementById('votingMode');
  const regionObjectsGroup = document.getElementById('regionObjectsGroup');
  const answerObjectsGroup = document.getElementById('answerObjectsGroup');
  const regionAnswerSlotGroup = document.getElementById('regionAnswerSlotGroup');
  const regionsModeHint = document.getElementById('regionsModeHint');

  if (votingModeInput && votingModeInput.value !== mode) {
    votingModeInput.value = mode;
  }
  if (regionObjectsGroup) {
    regionObjectsGroup.classList.toggle('config-hidden', isObjectMode);
  }
  if (answerObjectsGroup) {
    answerObjectsGroup.classList.toggle('config-hidden', !isObjectMode);
  }
  if (regionAnswerSlotGroup) {
    regionAnswerSlotGroup.classList.toggle('config-hidden', isObjectMode);
  }
  if (regionsModeHint) {
    regionsModeHint.textContent = isObjectMode
      ? t('regions.objectModeHint')
      : t('regions.regionModeHint');
  }
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

function initializeShowDetectionAnswerControl() {
  const input = document.getElementById('showDetectionAnswer');
  if (!input) return;

  try {
    const stored = localStorage.getItem(SHOW_DETECTION_ANSWER_STORAGE_KEY);
    if (stored !== null) {
      showDetectionAnswer = stored !== '0';
    }
  } catch (_err) { /* ignore */ }

  input.checked = showDetectionAnswer;
  input.addEventListener('change', () => {
    showDetectionAnswer = input.checked;
    try {
      localStorage.setItem(SHOW_DETECTION_ANSWER_STORAGE_KEY, showDetectionAnswer ? '1' : '0');
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
    return `${t(normalizedSlot === 2 ? 'common.answer2' : 'common.answer1')} (${answerText})`;
  }
  return t(normalizedSlot === 2 ? 'common.answer2' : 'common.answer1');
}

function getAnswerSlotForResultText(lastResult, text) {
  if (!lastResult || !text) return null;
  const answers = lastResult.question?.answers;
  if (!Array.isArray(answers)) return null;
  const idx = answers.indexOf(text);
  return idx === -1 ? null : idx + 1;
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
  const votingData = state?.voting || {};
  const lastResult = votingData.phase === 'pause' ? votingData.last_vote_result : null;
  const correctSlot = getAnswerSlotForResultText(lastResult, lastResult?.question?.correct_answer);

  (state.assignments || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.box;
    const [dx1, dy1] = toDisplayPoint([x1, y1]);
    const [dx2, dy2] = toDisplayPoint([x2, y2]);
    const dx = Math.min(dx1, dx2);
    const dy = Math.min(dy1, dy2);
    const dw = Math.abs(dx2 - dx1);
    const dh = Math.abs(dy2 - dy1);
    const voteSlot = Number((det.vote_slot ?? det.region_answer_slot) || 0);
    const color = voteSlot ? getSlotColor(voteSlot) : '#b9c0ce';
    const assignedText = describeAssignment(det);
    const labelPart = `${det.label} ${Number(det.score || 0).toFixed(2)}`;
    let caption = '';
    if (showDetectionLabel && showDetectionAnswer) {
      caption = `${labelPart} -> ${assignedText}`;
    } else if (showDetectionLabel) {
      caption = labelPart;
    } else if (showDetectionAnswer) {
      caption = assignedText;
    }

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.5, Math.min(3, displayWidth / 640));
    if (correctSlot && voteSlot) {
      ctx.fillStyle = voteSlot === correctSlot ? 'rgba(34, 197, 94, 0.45)' : 'rgba(239, 68, 68, 0.45)';
      ctx.fillRect(dx, dy, dw, dh);
    }
    ctx.strokeRect(dx, dy, dw, dh);

    if (caption) {
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
    }
    ctx.restore();
  });
}

function drawOverlay() {
  ctx.clearRect(0, 0, getCanvasDisplayWidth(), getCanvasDisplayHeight());
  drawDetections();
  regions.forEach((r, index) => {
    const isSelected = Number(r.id) === Number(selectedRegionId);
    drawPolygon(r.points, getRegionColor(r, index, isSelected), false, isSelected);
  });
}

function refreshRegionList() {
  const host = document.getElementById('regionList');
  if (!regions.length) {
    host.textContent = t('common.noRegions');
    selectedRegionId = null;
    syncRegionEditorFromSelected();
    return;
  }
  if (!regions.some((r) => Number(r.id) === Number(selectedRegionId))) {
    selectedRegionId = Number(regions[0].id);
  }
  host.innerHTML = regions.map((r) => {
    const cls = Number(r.id) === Number(selectedRegionId) ? 'region-item selected' : 'region-item';
    const slotMeta = isObjectListVotingMode()
      ? t('common.voteArea')
      : getAnswerSlotName(r.answer_slot || 1);
    const criterionLabel = r.criterion === 'inside' ? t('regions.fullyInside') : t('regions.overlap');
    return `<div class="${cls}" data-id="${r.id}">#${r.id} <b>${slotMeta}</b><span class="region-meta">${criterionLabel} - ${r.points.length} pts</span></div>`;
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
  if (!chart) return;
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
    host.innerHTML = `<div class="chip">${escapeHtml(t('common.noLabeledCounts'))}</div>`;
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
  const countdownPhaseDisplay = document.getElementById('countdownPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');
  const votingTimer = document.getElementById('votingTimer');
  const votingTimeLeft = document.getElementById('votingTimeLeft');
  const votingTimerPrompt = document.getElementById('votingTimerPrompt');

  // Hide other displays
  countdownPhaseDisplay.style.display = 'none';
  pausePhaseDisplay.style.display = 'none';
  idlePhaseDisplay.style.display = 'none';
  votingPhaseDisplay.style.display = 'block';

  // Update question
  const question = votingData.current_question;
  if (question) {
    currentQuestionSource = question.source || null;
    setMarkdownContent('votingQuestion', question.question, '');

    // Update answer blocks
    const answers = question.answers || [];
    const count1 = Number(slotCounts?.[1] || slotCounts?.['1'] || 0);
    const count2 = Number(slotCounts?.[2] || slotCounts?.['2'] || 0);

    // Update Answer 1
    setMarkdownContent('answerText1', answers[0] || '', t('common.answer1'));
    document.getElementById('answerCount1').textContent = count1;

    // Update Answer 2
    setMarkdownContent('answerText2', answers[1] || '', t('common.answer2'));
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
    const isUrgent = timeLeft <= 5;
    const promptKey = getVotingMode() === 'object_lists'
      ? 'voting.showYourVote'
      : 'voting.chooseYourVote';
    votingTimeLeft.textContent = t('voting.timeLeft', { value: timeLeft });
    votingTimerPrompt.textContent = t(promptKey);
    votingTimer.classList.toggle('urgent', isUrgent);
    votingTimerPrompt.hidden = !isUrgent;
  }
}

function updateCountdownPhaseDisplay(votingData) {
  const votingPhaseDisplay = document.getElementById('votingPhaseDisplay');
  const countdownPhaseDisplay = document.getElementById('countdownPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');
  const countChipsHost = document.getElementById('countChips');
  const voteStatsHost = document.getElementById('voteStats');
  const votingTimer = document.getElementById('votingTimer');
  const countdownTip = document.getElementById('countdownTip');

  // Hide other displays
  votingPhaseDisplay.style.display = 'none';
  pausePhaseDisplay.style.display = 'none';
  idlePhaseDisplay.style.display = 'none';
  countdownPhaseDisplay.style.display = 'block';
  if (countChipsHost) {
    countChipsHost.style.display = 'none';
  }
  if (voteStatsHost) {
    voteStatsHost.style.display = 'none';
  }
  if (votingTimer) {
    votingTimer.classList.remove('urgent');
  }

  const countdownValue = Math.max(1, Math.ceil(Number(votingData.time_left_sec ?? 0)));
  document.getElementById('countdownNumber').textContent = String(countdownValue);
  if (countdownTipKey === null) {
    const availableTipKeys = COUNTDOWN_TIP_KEYS.filter((key) => key !== previousCountdownTipKey);
    countdownTipKey = availableTipKeys[Math.floor(Math.random() * availableTipKeys.length)]
      || COUNTDOWN_TIP_KEYS[0];
    previousCountdownTipKey = countdownTipKey;
  }
  countdownTip.textContent = t(countdownTipKey);
}

function updatePausePhaseDisplay(votingData, slotCounts) {
  const votingPhaseDisplay = document.getElementById('votingPhaseDisplay');
  const countdownPhaseDisplay = document.getElementById('countdownPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');
  const votingTimer = document.getElementById('votingTimer');

  // Hide other displays
  votingPhaseDisplay.style.display = 'none';
  countdownPhaseDisplay.style.display = 'none';
  idlePhaseDisplay.style.display = 'none';
  pausePhaseDisplay.style.display = 'block';
  if (votingTimer) {
    votingTimer.classList.remove('urgent');
  }

  const lastResult = votingData.last_vote_result;
  if (lastResult) {
    const question = lastResult.question;
    currentQuestionSource = question?.source || null;

    // Update question
    setMarkdownContent('pauseQuestionText', question.question, '');

    // Update counts
    const answers = question.answers || [];
    const count1 = lastResult.counts?.[answers[0]] || 0;
    const count2 = lastResult.counts?.[answers[1]] || 0;

    setMarkdownContent('pauseAnswer1Label', answers[0] || '', t('common.answer1'));
    document.getElementById('pauseAnswer1Count').textContent = count1;
    setMarkdownContent('pauseAnswer2Label', answers[1] || '', t('common.answer2'));
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
      resultMsg.textContent = t('common.tie');
      resultMsg.className = 'pause-result-message tie';
    } else if (lastResult.is_correct === true) {
      resultMsg.textContent = t('common.correct');
      resultMsg.className = 'pause-result-message correct';
    } else if (lastResult.is_correct === false) {
      resultMsg.textContent = t('common.wrong');
      resultMsg.className = 'pause-result-message incorrect';
    } else {
      resultMsg.textContent = t('common.resultUnknown');
      resultMsg.className = 'pause-result-message tie';
    }
    
    // Apply check/cross marks to answer items
    const correctAnswer = question.correct_answer || t('common.notSpecified');
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
    correctWrongDiv.innerHTML = `${escapeHtml(t('common.correctAnswer'))} ${renderBasicMarkdown(correctAnswer) || escapeHtml(t('common.notSpecified'))}${moreInfo}`;
    
    // Update accuracy display
    const accuracyContainer = document.getElementById('pauseAccuracyContainer');
    const recentVotes = votingData.recent_scored_votes || 0;
    if (recentVotes > 0 && votingData.majority_vote_accuracy_percent !== null) {
      accuracyContainer.style.display = 'block';
      const accuracy = votingData.majority_vote_accuracy_percent;
      document.getElementById('pauseAccuracyLabel').textContent = t('common.recentMajorityVoteAccuracyQuestions', { value: recentVotes });
      document.getElementById('pauseAccuracyValue').textContent = `${accuracy}%`;
    } else {
      accuracyContainer.style.display = 'none';
    }

    const singleVoteAccuracyContainer = document.getElementById('pauseSingleVoteAccuracyContainer');
    if (recentVotes > 0 && votingData.single_vote_accuracy_percent !== null) {
      singleVoteAccuracyContainer.style.display = 'block';
      const singleVoteAccuracy = votingData.single_vote_accuracy_percent;
      document.getElementById('pauseSingleVoteAccuracyLabel').textContent = t('common.recentSingleVoteAccuracyQuestions', { value: recentVotes });
      document.getElementById('pauseSingleVoteAccuracyValue').textContent = `${singleVoteAccuracy}%`;
    } else {
      singleVoteAccuracyContainer.style.display = 'none';
    }
    
    // Update countdown
    const timeLeft = Number(votingData.time_left_sec ?? 0);
    document.getElementById('pauseCountdown').textContent = t('common.nextQuestion', { value: timeLeft });
  }
}

function updateVotingDisplay(votingData, slotCounts) {
  const votingPhaseDisplay = document.getElementById('votingPhaseDisplay');
  const countdownPhaseDisplay = document.getElementById('countdownPhaseDisplay');
  const pausePhaseDisplay = document.getElementById('pausePhaseDisplay');
  const idlePhaseDisplay = document.getElementById('idlePhaseDisplay');
  const votingTimer = document.getElementById('votingTimer');

  if (!votingData.active || votingData.phase !== 'countdown') {
    countdownTipKey = null;
  }

  if (!votingData.active) {
    currentQuestionSource = null;
    // Show idle display
    votingPhaseDisplay.style.display = 'none';
    countdownPhaseDisplay.style.display = 'none';
    pausePhaseDisplay.style.display = 'none';
    idlePhaseDisplay.style.display = 'block';
    if (votingTimer) {
      votingTimer.classList.remove('urgent');
    }
    return;
  }

  if (votingData.phase === 'countdown') {
    updateCountdownPhaseDisplay(votingData);
  } else if (votingData.phase === 'question') {
    updateVotingPhaseDisplay(votingData, slotCounts);
  } else if (votingData.phase === 'pause') {
    updatePausePhaseDisplay(votingData, slotCounts);
  } else {
    // Fallback to idle
    votingPhaseDisplay.style.display = 'none';
    countdownPhaseDisplay.style.display = 'none';
    pausePhaseDisplay.style.display = 'none';
    idlePhaseDisplay.style.display = 'block';
    if (votingTimer) {
      votingTimer.classList.remove('urgent');
    }
  }
}

function refreshVoting(v) {
  const mappingModeLabel = getVotingMode() === 'object_lists' ? t('common.objectLists') : t('common.regionSlots');
  document.getElementById('modeLabel').textContent = `${t('common.mode')}: ${v.active ? t('common.voting') : t('common.counting')} / ${mappingModeLabel}`;
  const countChipsHost = document.getElementById('countChips');
  countChipsHost.style.display = v.active ? 'none' : 'flex';

  const voteDurationInput = document.getElementById('voteDuration');
  const pauseDurationInput = document.getElementById('pauseDuration');
  const preQuestionCountdownInput = document.getElementById('preQuestionCountdown');
  const rollingWindowInput = document.getElementById('rollingWindow');
  const addReadingTimeInput = document.getElementById('addReadingTime');
  const shuffleAnswersInput = document.getElementById('shuffleAnswers');
  if (document.activeElement !== voteDurationInput) voteDurationInput.value = String(v.vote_duration_sec ?? voteDurationInput.value);
  if (document.activeElement !== pauseDurationInput) pauseDurationInput.value = String(v.pause_duration_sec ?? pauseDurationInput.value);
  if (document.activeElement !== preQuestionCountdownInput) preQuestionCountdownInput.value = String(v.pre_question_countdown_sec ?? preQuestionCountdownInput.value);
  if (document.activeElement !== rollingWindowInput) rollingWindowInput.value = String(v.window_size ?? rollingWindowInput.value);
  if (addReadingTimeInput && document.activeElement !== addReadingTimeInput) {
    addReadingTimeInput.checked = Boolean(v.add_reading_time ?? false);
  }
  if (shuffleAnswersInput && document.activeElement !== shuffleAnswersInput) {
    shuffleAnswersInput.checked = Boolean(v.shuffle_answers ?? true);
  }
  lastVotingConfigSignature = `${Number(v.vote_duration_sec || 0)}|${Number(v.pause_duration_sec || 0)}|${Number(v.pre_question_countdown_sec || 0)}|${Number(v.window_size || 0)}|${Boolean(v.add_reading_time ?? false)}|${Boolean(v.shuffle_answers ?? true)}`;

  // Update voting display
  const slotCounts = state?.slot_counts || {};
  updateVotingDisplay(v, slotCounts);

  // Keep the stats for idle phase
  const statHost = document.getElementById('voteStats');
  const majorityAcc = v.majority_vote_accuracy_percent === null ? 'n/a' : `${v.majority_vote_accuracy_percent}%`;
  const singleAcc = v.single_vote_accuracy_percent === null ? 'n/a' : `${v.single_vote_accuracy_percent}%`;
  statHost.innerHTML = [
    `<div class="chip">${t('common.majorityVoteAccuracy')}: ${majorityAcc}</div>`,
    `<div class="chip">${t('common.singleVoteAccuracy')}: ${singleAcc}</div>`,
    `<div class="chip">${t('common.scoredVotes')}: ${v.recent_scored_votes}</div>`,
  ].join('');
  statHost.style.display = v.active ? 'none' : 'flex';
}

function readDetectorDraftFromInputs() {
  const votingMode = document.getElementById('votingMode').value || 'region_slots';
  const detector = document.getElementById('detectorType').value;
  const modelName = document.getElementById('modelNameInput').value.trim() || null;
  const regionObjects = parseTextareaList(document.getElementById('objectsInput').value);
  const answer1Objects = parseTextareaList(document.getElementById('answer1ObjectsInput').value);
  const answer2Objects = parseTextareaList(document.getElementById('answer2ObjectsInput').value);
  const threshold = Number(document.getElementById('thresholdInput').value);
  const frameSkip = Number(document.getElementById('frameSkipInput').value);

  if (!detector) {
    return null;
  }

  return {
    detector,
    model_name: modelName,
    voting_mode: votingMode,
    region_objects: regionObjects,
    answer_objects: {
      1: answer1Objects,
      2: answer2Objects,
    },
    threshold,
    frame_skip: Math.trunc(frameSkip),
  };
}

function getDetectorSignature(payload) {
  return [
    payload.detector,
    payload.model_name || '',
    payload.voting_mode,
    payload.threshold,
    payload.frame_skip,
    getObjectListSignature(payload.region_objects || []),
    getObjectListSignature(payload.answer_objects?.[1] || payload.answer_objects?.['1'] || []),
    getObjectListSignature(payload.answer_objects?.[2] || payload.answer_objects?.['2'] || []),
  ].join('|');
}

function isValidDetectorPayload(payload) {
  if (!payload || !payload.detector || !Number.isFinite(payload.threshold) || !Number.isFinite(payload.frame_skip) || payload.frame_skip < 1) {
    return false;
  }

  if (payload.voting_mode === 'object_lists') {
    const answer1Objects = payload.answer_objects?.[1] || payload.answer_objects?.['1'] || [];
    const answer2Objects = payload.answer_objects?.[2] || payload.answer_objects?.['2'] || [];
    if (!answer1Objects.length || !answer2Objects.length) {
      return false;
    }
    const overlap = new Set(answer1Objects.map((item) => item.toLowerCase()));
    if (answer2Objects.some((item) => overlap.has(item.toLowerCase()))) {
      return false;
    }
    return true;
  }

  return Array.isArray(payload.region_objects) && payload.region_objects.length > 0;
}

function syncDetectorInputsFromState(nextState) {
  if (!nextState || !nextState.detector) return;
  const votingModeInput = document.getElementById('votingMode');
  const detectorTypeInput = document.getElementById('detectorType');
  const modelNameInput = document.getElementById('modelNameInput');
  const objectsInput = document.getElementById('objectsInput');
  const answer1ObjectsInput = document.getElementById('answer1ObjectsInput');
  const answer2ObjectsInput = document.getElementById('answer2ObjectsInput');
  const thresholdInput = document.getElementById('thresholdInput');
  const frameSkipInput = document.getElementById('frameSkipInput');

  const serverPayload = {
    detector: nextState.detector.type || detectorTypeInput.value,
    model_name: nextState.detector.model_name || null,
    voting_mode: nextState.detector.voting_mode || votingModeInput.value || 'region_slots',
    region_objects: Array.isArray(nextState.detector.region_objects) ? nextState.detector.region_objects : [],
    answer_objects: {
      1: Array.isArray(nextState.detector.answer_objects?.['1']) ? nextState.detector.answer_objects['1'] : [],
      2: Array.isArray(nextState.detector.answer_objects?.['2']) ? nextState.detector.answer_objects['2'] : [],
    },
    threshold: Number(nextState.detector.threshold ?? thresholdInput.value),
    frame_skip: Math.trunc(Number(nextState.detector.frame_skip ?? frameSkipInput.value)),
  };
  const serverSignature = getDetectorSignature(serverPayload);
  if (isValidDetectorPayload(serverPayload)) {
    lastDetectorSignature = serverSignature;
  }

  const localDraft = readDetectorDraftFromInputs();
  const localSignature = localDraft ? getDetectorSignature(localDraft) : null;
  const detectorConfigDirty = !!(detectorInputsInitialized && localSignature && localSignature !== lastDetectorSignature);
  if (detectorConfigDirty) {
    updateVotingModeUi();
    return;
  }

  if (document.activeElement !== votingModeInput) {
    votingModeInput.value = nextState.detector.voting_mode || votingModeInput.value;
  }
  if (document.activeElement !== detectorTypeInput) {
    detectorTypeInput.value = nextState.detector.type || detectorTypeInput.value;
  }
  if (document.activeElement !== modelNameInput) {
    modelNameInput.value = nextState.detector.model_name || '';
  }
  if (document.activeElement !== objectsInput) {
    objectsInput.value = (nextState.detector.region_objects || []).join('\n');
  }
  if (document.activeElement !== answer1ObjectsInput) {
    answer1ObjectsInput.value = (nextState.detector.answer_objects?.['1'] || []).join('\n');
  }
  if (document.activeElement !== answer2ObjectsInput) {
    answer2ObjectsInput.value = (nextState.detector.answer_objects?.['2'] || []).join('\n');
  }
  if (document.activeElement !== thresholdInput) {
    thresholdInput.value = String(nextState.detector.threshold ?? thresholdInput.value);
  }
  if (document.activeElement !== frameSkipInput) {
    frameSkipInput.value = String(nextState.detector.frame_skip ?? frameSkipInput.value);
  }
  updateVotingModeUi(nextState.detector.voting_mode || votingModeInput.value);
  detectorInputsInitialized = true;
}

async function saveDetectorFromInputs() {
  const payload = readDetectorDraftFromInputs();
  if (!isValidDetectorPayload(payload)) {
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
  const preQuestionCountdown = Number(document.getElementById('preQuestionCountdown').value);
  const windowSize = Number(document.getElementById('rollingWindow').value);
  const addReadingTime = !!document.getElementById('addReadingTime')?.checked;
  const shuffleAnswers = !!document.getElementById('shuffleAnswers')?.checked;

  if (!Number.isFinite(voteDuration) || !Number.isFinite(pauseDuration) || !Number.isFinite(preQuestionCountdown) || !Number.isFinite(windowSize)) {
    return null;
  }

  const payload = {
    vote_duration_sec: Math.trunc(voteDuration),
    pause_duration_sec: Math.trunc(pauseDuration),
    pre_question_countdown_sec: Math.trunc(preQuestionCountdown),
    window_size: Math.trunc(windowSize),
    add_reading_time: addReadingTime,
    shuffle_answers: shuffleAnswers
  };

  if (payload.vote_duration_sec < 1 || payload.pause_duration_sec < 0 || payload.pre_question_countdown_sec < 0 || payload.window_size < 1) {
    return null;
  }

  return payload;
}

function getVotingConfigSignature(payload) {
  return `${payload.vote_duration_sec}|${payload.pause_duration_sec}|${payload.pre_question_countdown_sec}|${payload.window_size}|${Boolean(payload.add_reading_time)}|${Boolean(payload.shuffle_answers)}`;
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

function normalizeQuestionSources(rawSources) {
  if (!Array.isArray(rawSources)) {
    return [];
  }

  return rawSources
    .map((entry) => {
      const name = String(entry?.name || '').trim();
      if (!name) return null;
      return {
        name,
        selected: entry?.selected !== false,
      };
    })
    .filter(Boolean);
}

function getSelectedQuestionSources() {
  return questionSources.filter((entry) => entry.selected).map((entry) => entry.name);
}

function syncQuestionSourcesFromState(nextState) {
  const serverSelected = nextState?.question_sources?.selected;
  if (!Array.isArray(serverSelected) || !questionSources.length) return;
  if (questionLoadInFlight || questionLoadQueued) return;

  const host = document.getElementById('questionSources');
  if (host && host.contains(document.activeElement)) return;

  const serverSet = new Set(serverSelected);
  const localSet = new Set(getSelectedQuestionSources());
  const isSameSelection = serverSet.size === localSet.size && [...serverSet].every((name) => localSet.has(name));
  if (isSameSelection) return;

  questionSources.forEach((entry) => {
    entry.selected = serverSet.has(entry.name);
  });
  renderQuestionSources();
}

function renderQuestionSources() {
  const host = document.getElementById('questionSources');
  if (!host) return;

  if (!questionSources.length) {
    host.innerHTML = '<div class="question-source-empty">No .jsonl/.json/.zip files found in data/.</div>';
    return;
  }

  host.innerHTML = questionSources
    .map((entry, idx) => {
      const checked = entry.selected ? 'checked' : '';
      const safeName = escapeHtml(entry.name);
      return `<label class="question-source-item"><input type="checkbox" data-question-source-index="${idx}" ${checked} /><span>${safeName}</span></label>`;
    })
    .join('');

  Array.from(host.querySelectorAll('input[data-question-source-index]')).forEach((input) => {
    input.addEventListener('change', () => {
      const index = Number(input.dataset.questionSourceIndex);
      if (!Number.isInteger(index) || index < 0 || index >= questionSources.length) {
        return;
      }

      const nextSelected = !!input.checked;
      if (!nextSelected) {
        const selectedCount = questionSources.filter((entry) => entry.selected).length;
        if (selectedCount <= 1) {
          input.checked = true;
          flash('At least one question file must remain selected.');
          return;
        }
      }

      questionSources[index].selected = nextSelected;
      queueLoadQuestionsFromSelection(true);
    });
  });
}

async function loadQuestionsFromSelection({ silentSuccess = false } = {}) {
  const selectedSources = getSelectedQuestionSources();
  if (!selectedSources.length) {
    flash('Select at least one question file first.');
    return;
  }

  if (questionLoadInFlight) {
    questionLoadQueued = true;
    return;
  }

  questionLoadInFlight = true;
  try {
    const formData = new FormData();
    selectedSources.forEach((name) => formData.append('selected_files', name));

    const uploadInput = document.getElementById('questionsFile');
    const uploadedFile = uploadInput?.files?.[0] || null;
    if (uploadedFile) {
      formData.append('questions_file', uploadedFile, uploadedFile.name);
    }

    const resp = await fetch('/voting_questions', {
      method: 'POST',
      body: formData
    });
    const data = await readJsonResponse(resp, `Unexpected response from ${resp.url}`);

    if (!data || data.status !== 'success') {
      flash(data?.message || 'Questions update failed');
      return;
    }

    if (!silentSuccess) {
      flash(data.message || 'Questions updated');
    }
    if (uploadInput) {
      uploadInput.value = '';
    }
  } catch (error) {
    flash(`Questions update failed: ${error.message}`);
  } finally {
    questionLoadInFlight = false;
    if (questionLoadQueued) {
      questionLoadQueued = false;
      await loadQuestionsFromSelection({ silentSuccess: true });
    }
  }
}

function queueLoadQuestionsFromSelection(silentSuccess = true) {
  window.setTimeout(() => {
    loadQuestionsFromSelection({ silentSuccess });
  }, 0);
}

async function readJsonResponse(resp, fallbackMessage) {
  const contentType = String(resp.headers.get('content-type') || '').toLowerCase();
  if (contentType.includes('application/json')) {
    return resp.json();
  }

  const text = await resp.text();
  const compact = text.replace(/\s+/g, ' ').trim();
  const preview = compact.slice(0, 160) || fallbackMessage;
  throw new Error(fallbackMessage ? `${fallbackMessage}: ${preview}` : preview);
}

async function fetchQuestionSources() {
  const host = document.getElementById('questionSources');
  if (host) {
    host.innerHTML = 'Loading files...';
  }

  try {
    const resp = await fetch('/voting_question_sources');
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await readJsonResponse(resp, `Unexpected response from ${resp.url}`);
    if (!data || data.status !== 'success') {
      throw new Error(data?.message || 'Failed to load question file list');
    }

    questionSources = normalizeQuestionSources(data.sources);
    if (questionSources.length && !questionSources.some((entry) => entry.selected)) {
      questionSources[0].selected = true;
    }
    renderQuestionSources();
  } catch (error) {
    questionSources = [];
    if (host) {
      host.innerHTML = '<div class="question-source-empty">Could not load question files.</div>';
    }
    flash(`Question source load failed: ${error.message}`);
  }
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
  syncQuestionSourcesFromState(nextState);
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

function normalizedRegionsForConfiguration() {
  const width = Number(state?.frame?.width);
  const height = Number(state?.frame?.height);
  if (!(width > 0) || !(height > 0)) {
    throw new Error('Camera frame dimensions are not available yet');
  }
  return regions.map((region) => ({
    id: Number(region.id),
    answer_slot: Number(region.answer_slot || 1),
    criterion: region.criterion || 'overlap',
    points: (region.points || []).map(([x, y]) => [
      Number((Number(x) / Math.max(1, width - 1)).toFixed(8)),
      Number((Number(y) / Math.max(1, height - 1)).toFixed(8)),
    ]),
  }));
}

function buildConfigurationPayload() {
  const detector = readDetectorDraftFromInputs();
  const voting = readVotingConfigPayloadFromInputs();
  if (!isValidDetectorPayload(detector) || !voting) {
    throw new Error('Correct invalid detector or voting values before saving');
  }

  return {
    format: 'detect-and-vote-configuration',
    version: 1,
    detector,
    regions: {
      coordinate_space: 'normalized',
      items: normalizedRegionsForConfiguration(),
    },
    slot_colors: {
      1: document.getElementById('answer1Color').value,
      2: document.getElementById('answer2Color').value,
    },
    voting,
    web_ui: {
      update_interval_ms: updateIntervalMs,
      mirror_view: mirrorEnabled,
      show_detection_label: showDetectionLabel,
      show_detection_answer: showDetectionAnswer,
      panel_visibility: collectPanelVisibilityFromInputs(),
    },
  };
}

function downloadConfiguration() {
  const payload = buildConfigurationPayload();
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: 'application/json' });
  const link = document.createElement('a');
  const date = new Date().toISOString().slice(0, 10);
  link.href = URL.createObjectURL(blob);
  link.download = `detect-and-vote-config-${date}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function postConfigurationPart(url, payload, label) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await readJsonResponse(resp, `${label} returned an invalid response`);
  if (!resp.ok || data?.status !== 'success') {
    throw new Error(data?.message || `${label} failed with HTTP ${resp.status}`);
  }
  return data;
}

function regionsFromConfiguration(regionConfig) {
  if (!regionConfig || regionConfig.coordinate_space !== 'normalized' || !Array.isArray(regionConfig.items)) {
    throw new Error('The configuration has no valid normalized answer regions');
  }
  const width = Number(state?.frame?.width);
  const height = Number(state?.frame?.height);
  if (!(width > 0) || !(height > 0)) {
    throw new Error('Camera frame dimensions are not available yet');
  }

  const seenIds = new Set();
  return regionConfig.items.map((region) => {
    const id = Number(region.id);
    const answerSlot = Number(region.answer_slot);
    const criterion = String(region.criterion || '');
    if (!Number.isInteger(id) || id < 1 || seenIds.has(id)) {
      throw new Error(`Region ${region.id ?? '?'} has an invalid or duplicate id`);
    }
    seenIds.add(id);
    if (![1, 2].includes(answerSlot) || !['inside', 'overlap'].includes(criterion)) {
      throw new Error(`Region ${id} has an invalid answer slot or criterion`);
    }
    if (!Array.isArray(region.points) || region.points.length < 3) {
      throw new Error(`Region ${id} has fewer than three points`);
    }
    const points = region.points.map((point) => {
      const x = Number(point?.[0]);
      const y = Number(point?.[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) {
        throw new Error(`Region ${id} contains an invalid normalized point`);
      }
      return [
        Math.max(0, Math.min(width - 1, Math.round(x * Math.max(1, width - 1)))),
        Math.max(0, Math.min(height - 1, Math.round(y * Math.max(1, height - 1)))),
      ];
    });
    return {
      id,
      answer_slot: answerSlot,
      criterion,
      points,
    };
  });
}

function applyWebUiConfiguration(webUi = {}) {
  const interval = Math.max(100, Math.min(5000, Math.trunc(Number(webUi.update_interval_ms)) || 650));
  updateIntervalMs = interval;
  document.getElementById('updateIntervalInput').value = String(interval);
  try { localStorage.setItem(UPDATE_INTERVAL_STORAGE_KEY, String(interval)); } catch (_err) { /* ignore */ }

  setMirrorEnabled(Boolean(webUi.mirror_view));
  showDetectionLabel = webUi.show_detection_label !== false;
  document.getElementById('showDetectionLabel').checked = showDetectionLabel;
  try { localStorage.setItem(SHOW_DETECTION_LABEL_STORAGE_KEY, showDetectionLabel ? '1' : '0'); } catch (_err) { /* ignore */ }
  showDetectionAnswer = webUi.show_detection_answer !== false;
  document.getElementById('showDetectionAnswer').checked = showDetectionAnswer;
  try { localStorage.setItem(SHOW_DETECTION_ANSWER_STORAGE_KEY, showDetectionAnswer ? '1' : '0'); } catch (_err) { /* ignore */ }
  if (webUi.panel_visibility && typeof webUi.panel_visibility === 'object') {
    applyPanelVisibility(webUi.panel_visibility);
  }
}

async function applyConfiguration(payload) {
  if (!payload || payload.format !== 'detect-and-vote-configuration' || payload.version !== 1) {
    throw new Error('This is not a supported Detect & Vote configuration file');
  }
  if (!payload.detector || !payload.voting || !payload.slot_colors) {
    throw new Error('The configuration is missing required settings');
  }
  if (!isValidDetectorPayload(payload.detector) || !['owlvit', 'owlv2'].includes(payload.detector.detector)) {
    throw new Error('The configuration contains invalid detector settings');
  }
  const threshold = Number(payload.detector.threshold);
  if (threshold < 0 || threshold > 1) {
    throw new Error('The detector threshold must be between 0 and 1');
  }
  const voting = payload.voting;
  if (![voting.vote_duration_sec, voting.pause_duration_sec, voting.pre_question_countdown_sec, voting.window_size].every(Number.isInteger)
      || voting.vote_duration_sec <= 5 || voting.pause_duration_sec < 0
      || voting.pre_question_countdown_sec < 0 || voting.window_size < 1) {
    throw new Error('The configuration contains invalid voting timing values');
  }
  for (const slot of ['1', '2']) {
    if (!/^#[0-9a-f]{6}$/i.test(String(payload.slot_colors[slot] || ''))) {
      throw new Error(`Answer ${slot} has an invalid color`);
    }
  }

  const importedRegions = regionsFromConfiguration(payload.regions);
  await postConfigurationPart('/apply_settings', payload.detector, 'Detector settings');
  const regionResult = await postConfigurationPart('/set_regions', { regions: importedRegions }, 'Answer regions');
  await postConfigurationPart('/set_slot_colors', { slot_colors: payload.slot_colors }, 'Answer colors');
  await postConfigurationPart('/voting_config', payload.voting, 'Voting settings');

  setRegions(regionResult.regions || importedRegions);
  regionsDirty = false;
  applyWebUiConfiguration(payload.web_ui || {});
  detectorInputsInitialized = false;
  lastVotingConfigSignature = null;
  await fetchState();
  flash('Configuration loaded. Question datasets were left unchanged.');
}

document.getElementById('saveConfiguration').addEventListener('click', () => {
  try {
    downloadConfiguration();
    flash('Configuration JSON saved.');
  } catch (error) {
    flash(`Configuration save failed: ${error.message}`);
  }
});

document.getElementById('loadConfiguration').addEventListener('click', () => {
  document.getElementById('configurationFile').click();
});

document.getElementById('configurationFile').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    await applyConfiguration(payload);
  } catch (error) {
    flash(`Configuration load failed: ${error.message}`);
  }
});

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
  if (!isObjectListVotingMode()) {
    region.answer_slot = answerSlot;
  }
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

document.getElementById('votingMode').addEventListener('change', () => {
  if (isObjectListVotingMode()) {
    ensureObjectListDefaultsInInputs();
  }
  updateVotingModeUi();
  refreshRegionList();
  drawOverlay();
  queueDetectorSave();
});
document.getElementById('detectorType').addEventListener('change', queueDetectorSave);
document.getElementById('modelNameInput').addEventListener('change', queueDetectorSave);
document.getElementById('objectsInput').addEventListener('input', queueDetectorSave);
document.getElementById('objectsInput').addEventListener('change', queueDetectorSave);
document.getElementById('answer1ObjectsInput').addEventListener('input', queueDetectorSave);
document.getElementById('answer1ObjectsInput').addEventListener('change', queueDetectorSave);
document.getElementById('answer2ObjectsInput').addEventListener('input', queueDetectorSave);
document.getElementById('answer2ObjectsInput').addEventListener('change', queueDetectorSave);
document.getElementById('thresholdInput').addEventListener('input', queueDetectorSave);
document.getElementById('thresholdInput').addEventListener('change', queueDetectorSave);
document.getElementById('frameSkipInput').addEventListener('input', queueDetectorSave);
document.getElementById('frameSkipInput').addEventListener('change', queueDetectorSave);

['voteDuration', 'pauseDuration', 'preQuestionCountdown', 'rollingWindow'].forEach((id) => {
  const el = document.getElementById(id);
  el.addEventListener('input', queueVotingConfigSave);
  el.addEventListener('change', queueVotingConfigSave);
});
document.getElementById('addReadingTime').addEventListener('change', queueVotingConfigSave);
document.getElementById('shuffleAnswers').addEventListener('change', queueVotingConfigSave);

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
  await loadQuestionsFromSelection({ silentSuccess: false });
};

window.addEventListener('resize', () => scheduleResizeSync(12));
window.addEventListener('load', () => scheduleResizeSync(16));
video.addEventListener('load', () => scheduleResizeSync(16));

if (typeof ResizeObserver !== 'undefined') {
  const resizeObserver = new ResizeObserver(() => scheduleResizeSync(6));
  resizeObserver.observe(video);
  resizeObserver.observe(videoWrap);
}

initializeLanguageControl();
initializeShowDetectionLabelControl();
initializeShowDetectionAnswerControl();
initializeMirrorViewControl();
initializeUpdateIntervalControl();
initializePanelVisibilityControls();
initializePanelToolbarAutoHide();
updateVotingModeUi();
fetchQuestionSources();

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
