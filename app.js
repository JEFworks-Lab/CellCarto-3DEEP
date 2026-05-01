import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { parquetRead } from 'https://esm.sh/hyparquet';
import { compressors } from 'https://esm.sh/hyparquet-compressors';

// Parquet shard files (each shard = 10% random sample, pre-shuffled)
// Loading N shards = N * 10% sample rate
const PARQUET_SHARDS = [
    'data/hairfollicle.shard01.parquet',
    'data/hairfollicle.shard02.parquet',
    'data/hairfollicle.shard03.parquet',
    'data/hairfollicle.shard04.parquet',
    'data/hairfollicle.shard05.parquet',
    'data/hairfollicle.shard06.parquet',
    'data/hairfollicle.shard07.parquet',
    'data/hairfollicle.shard08.parquet',
    'data/hairfollicle.shard09.parquet',
    'data/hairfollicle.shard10.parquet',
];

// Configuration
const MAX_POINTS = 8000000; // Max points to render for performance (reduced from 2M)
const KO_TRANSFORMED_Y_OFFSET = -1000; // Shift KO follicles up in the transformed coordinate space so they share the same camera view as WT
const DEFAULT_TIMERANK = 260; // Initial TimeRank shown in slider/filter on load
    
// Column configuration - users can manually change these lists
const COORD_PRESETS = {
    transformed: { x: 'transformedX', y: 'transformedZ', z: 'transformedY' },
    raw:         { x: 'x',            y: 'z',            z: 'y' }
};
const COORD_COLUMN_OPTIONS = ['x', 'y', 'z', 'transformedX', 'transformedY', 'transformedZ', 'Adj_transformedZ', 'X_shifted', 'TimeRank'];
let selectedCoordX = COORD_PRESETS.transformed.x;
let selectedCoordY = COORD_PRESETS.transformed.y;
let selectedCoordZ = COORD_PRESETS.transformed.z;
const column_names_categorical = ['Structure', 'HF', 'Sample', 'Group', 'CellType', 'Gene'];
const UI_CATEGORICAL_OPTIONS = ['Structure', 'CellType', 'Gene'];
const column_names_continuous = ['Pseudotime'];

function getAttributeDisplayName(attribute) {
    return attribute === 'CellType' ? 'Cell-Type' : attribute;
}

// Global variables
let allData = [];
let visibleIndices = null; // Will be Uint32Array
let colorMap = new Map();
let attributeValues = {}; // Will be dynamically populated
let continuousRanges = {}; // Will store min/max for each continuous variable
let cameraInitialized = false;
let activeFilters = [];
let initialCameraState = { position: null, target: null };
let autoRotateEnabled = false;
let autoShiftEnabled = false;
let autoShiftIntervalId = null;
let autoShiftTimeRankValues = [];
let autoShiftCurrentIndex = { wt: 0, ko: 0 };
const AUTO_SHIFT_INTERVAL_MS = 500;
let timerankFilterActive = true;
let timerankCoupled = true;
let tooltip = null;
let isShiftPressed = false;
let eventListenersInitialized = false;

// Dual-viewport state: WT (top) and KO (bottom)
const viewports = {
    wt: { scene: null, camera: null, renderer: null, controls: null, pointCloud: null, renderedIndicesMap: null, highlightSphere: null, raycaster: new THREE.Raycaster(), mouse: new THREE.Vector2() },
    ko: { scene: null, camera: null, renderer: null, controls: null, pointCloud: null, renderedIndicesMap: null, highlightSphere: null, raycaster: new THREE.Raycaster(), mouse: new THREE.Vector2() }
};
let currentTheme = 'dark';

const THEME_STORAGE_KEY = 'cellcarto-theme';
const SCENE_THEME_COLORS = {
    dark: 0x1a1a1a,
    light: 0xf5f7fb
};

function updateSceneTheme(theme) {
    const color = SCENE_THEME_COLORS[theme] ?? SCENE_THEME_COLORS.dark;
    for (const key of ['wt', 'ko']) {
        if (viewports[key].scene) {
            viewports[key].scene.background = new THREE.Color(color);
        }
    }
}

function applyTheme(theme) {
    if (!document.body) {
        return;
    }
    currentTheme = theme;
    document.body.dataset.theme = theme;
    const toggle = document.getElementById('themeToggle');
    const label = document.querySelector('.theme-switch-text');
    if (toggle) {
        toggle.checked = theme === 'light';
    }
    if (label) {
        label.textContent = theme === 'light' ? 'Light' : 'Dark';
    }
    updateSceneTheme(theme);
}

function setupThemeToggle() {
    const toggle = document.getElementById('themeToggle');
    if (!toggle) {
        return;
    }
    const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    const prefersLight = window.matchMedia
        ? window.matchMedia('(prefers-color-scheme: light)').matches
        : false;
    const initialTheme = storedTheme || (prefersLight ? 'light' : 'dark');
    applyTheme(initialTheme);

    toggle.addEventListener('change', () => {
        const theme = toggle.checked ? 'light' : 'dark';
        applyTheme(theme);
        localStorage.setItem(THEME_STORAGE_KEY, theme);
    });
}

// Lazy loading state
let parquetBuffers = []; // Store downloaded buffers for lazy loading columns
let loadedColumns = new Set(); // Track which columns have been loaded
let isLoadingColumn = false; // Prevent concurrent column loads

// Progressive shard loading state
let loadedShardCount = 0; // How many shards are currently loaded
let isLoadingShards = false; // Prevent concurrent shard loads

// Helper: set up a single viewport (scene, camera, renderer, controls, highlight sphere)
function initViewport(key, canvasId, containerId) {
    const vp = viewports[key];
    const container = document.getElementById(containerId);
    const canvas = document.getElementById(canvasId);

    vp.scene = new THREE.Scene();
    updateSceneTheme(currentTheme);

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;
    vp.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 10000);
    vp.camera.position.set(0, 0, 100);

    vp.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    vp.renderer.setSize(width, height);
    vp.renderer.setPixelRatio(window.devicePixelRatio);

    vp.controls = new OrbitControls(vp.camera, vp.renderer.domElement);
    vp.controls.enableDamping = true;
    vp.controls.dampingFactor = 0.05;
    vp.controls.minDistance = 1;
    vp.controls.maxDistance = 10000;
    vp.controls.enableRotate = false;
    vp.controls.enablePan = false;
    vp.controls.enableZoom = false;
    vp.controls.screenSpacePanning = false;
    vp.controls.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: null };
    vp.controls.touches = { ONE: null, TWO: null };

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    vp.scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight.position.set(1, 1, 1);
    vp.scene.add(directionalLight);

    const highlightGeometry = new THREE.SphereGeometry(1, 16, 16);
    const highlightMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff, wireframe: true, side: THREE.DoubleSide,
        depthTest: true, depthWrite: false
    });
    vp.highlightSphere = new THREE.Mesh(highlightGeometry, highlightMaterial);
    vp.highlightSphere.renderOrder = 1000;
    vp.highlightSphere.visible = false;
    vp.scene.add(vp.highlightSphere);

    return vp;
}

// Initialize Two Three.js viewports (WT top, KO bottom)
function initScene() {
    initViewport('wt', 'scene-wt', 'viewport-wt');
    initViewport('ko', 'scene-ko', 'viewport-ko');

    // Shared drag / pan / rotate state across both viewports
    let isControlPressed = false;
    let isDragging = false;
    let lastMousePosition = new THREE.Vector2();
    const panSpeed = 0.1;
    const rotationSpeed = 0.01;

    window.addEventListener('keydown', (event) => {
        if (event.key === 'Control' || event.ctrlKey) isControlPressed = true;
    });
    window.addEventListener('keyup', (event) => {
        if (event.key === 'Control' || !event.ctrlKey) isControlPressed = false;
    });

    // Apply a camera delta to both viewports simultaneously
    function applyDelta(deltaX, deltaY) {
        for (const key of ['wt', 'ko']) {
            const vp = viewports[key];
            if (isControlPressed) {
                const offset = new THREE.Vector3().subVectors(vp.camera.position, vp.controls.target);
                if (Math.abs(deltaY) > 0) {
                    const m = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(1, 0, 0), deltaY * rotationSpeed);
                    offset.applyMatrix4(m);
                }
                if (Math.abs(deltaX) > 0) {
                    const m = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(0, 1, 0), deltaX * rotationSpeed);
                    offset.applyMatrix4(m);
                }
                vp.camera.position.copy(vp.controls.target).add(offset);
                vp.camera.lookAt(vp.controls.target);
            } else {
                const dist = vp.camera.position.distanceTo(vp.controls.target);
                const sp = panSpeed * (dist * 0.01);
                if (Math.abs(deltaY) > 0) {
                    const v = new THREE.Vector3(0, deltaY * sp, 0);
                    vp.controls.target.add(v);
                    vp.camera.position.add(v);
                }
                if (Math.abs(deltaX) > 0) {
                    const v = new THREE.Vector3(-deltaX * sp, 0, 0);
                    vp.controls.target.add(v);
                    vp.camera.position.add(v);
                }
            }
            vp.controls.update();
            vp.camera.updateMatrixWorld();
        }
    }

    // Attach drag handlers to the whole center panel so either viewport triggers pan/rotate
    const centerPanel = document.getElementById('center-panel');

    const handlePointerDown = (event) => {
        if (event.target.tagName !== 'CANVAS') return;
        if (event.button === 0 || event.button === 2 || event.buttons > 0) {
            event.preventDefault();
            isDragging = true;
            lastMousePosition.set(event.clientX, event.clientY);
        }
    };
    centerPanel.addEventListener('pointerdown', handlePointerDown, { capture: true, passive: false });

    const handlePointerMove = (event) => {
        if (!isDragging) return;
        event.preventDefault();
        const dx = event.clientX - lastMousePosition.x;
        const dy = event.clientY - lastMousePosition.y;
        applyDelta(dx, dy);
        lastMousePosition.set(event.clientX, event.clientY);
    };
    document.addEventListener('pointermove', handlePointerMove, { capture: true, passive: false });

    const handlePointerUp = () => { isDragging = false; };
    document.addEventListener('pointerup', handlePointerUp);
    document.addEventListener('pointercancel', handlePointerUp);

    // Zoom sync: apply identical zoom to both viewports
    centerPanel.addEventListener('wheel', (event) => {
        const delta = event.deltaY;
        const factor = 1 + delta * 0.001;
        for (const key of ['wt', 'ko']) {
            const vp = viewports[key];
            const offset = new THREE.Vector3().subVectors(vp.camera.position, vp.controls.target);
            offset.multiplyScalar(factor);
            vp.camera.position.copy(vp.controls.target).add(offset);
            vp.controls.update();
        }
        event.preventDefault();
    }, { passive: false });

    // Prevent context menu on right click when Control is held
    centerPanel.addEventListener('contextmenu', (event) => {
        if (isControlPressed) event.preventDefault();
    });

    window.addEventListener('resize', onWindowResize);

    // SHIFT key for hover highlighting
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Shift' || event.shiftKey) isShiftPressed = true;
    });
    window.addEventListener('keyup', (event) => {
        if (event.key === 'Shift' || !event.shiftKey) { isShiftPressed = false; hideHighlight(); }
    });

    // Shared tooltip
    tooltip = document.createElement('div');
    tooltip.id = 'point-tooltip';
    tooltip.className = 'point-tooltip';
    document.body.appendChild(tooltip);

    // Hover on each viewport
    for (const key of ['wt', 'ko']) {
        const canvas = viewports[key].renderer.domElement;
        canvas.addEventListener('pointermove', (event) => handleHoverForViewport(event, key), { passive: true });
    }
}

function onWindowResize() {
    for (const key of ['wt', 'ko']) {
        const vp = viewports[key];
        const container = vp.renderer.domElement.parentElement;
        if (!container) continue;
        const width = container.clientWidth;
        const height = container.clientHeight;
        vp.camera.aspect = width / height;
        vp.camera.updateProjectionMatrix();
        vp.renderer.setSize(width, height);
    }
}

function setupRightPanelResizer() {
    const container = document.getElementById('container');
    const resizer = document.getElementById('right-panel-resizer');
    const leftPanel = document.getElementById('left-panel');
    if (!container || !resizer || !leftPanel) {
        return;
    }

    const minRightWidth = 280;
    const minCenterWidth = 360;
    let dragging = false;
    let activePointerId = null;
    let resizeRafId = null;

    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

    const onPointerMove = (event) => {
        if (!dragging) return;
        const containerRect = container.getBoundingClientRect();
        const leftRect = leftPanel.getBoundingClientRect();
        const maxRightWidth = Math.max(
            minRightWidth,
            containerRect.width - leftRect.width - minCenterWidth
        );
        const proposed = containerRect.right - event.clientX;
        const width = clamp(proposed, minRightWidth, maxRightWidth);
        container.style.setProperty('--right-panel-width', `${Math.round(width)}px`);
        if (resizeRafId === null) {
            resizeRafId = requestAnimationFrame(() => {
                resizeRafId = null;
                onWindowResize();
            });
        }
    };

    const stopDragging = () => {
        if (!dragging) return;
        dragging = false;
        document.body.classList.remove('resizing');
        if (activePointerId !== null && resizer.releasePointerCapture) {
            try {
                resizer.releasePointerCapture(activePointerId);
            } catch (e) {
                // Ignore release errors on some browsers.
            }
        }
        activePointerId = null;
    };

    resizer.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        dragging = true;
        activePointerId = event.pointerId;
        if (resizer.setPointerCapture) {
            resizer.setPointerCapture(activePointerId);
        }
        document.body.classList.add('resizing');
        onPointerMove(event);
    });

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stopDragging);
    window.addEventListener('pointercancel', stopDragging);
}

// Handle hover for a specific viewport
function handleHoverForViewport(event, vpKey) {
    const vp = viewports[vpKey];
    if (!isShiftPressed || !vp.pointCloud || !vp.renderedIndicesMap) {
        hideHighlight();
        return;
    }
    if (event.buttons && event.buttons > 0) { hideHighlight(); return; }

    const canvas = vp.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    vp.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    vp.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const pointSize = parseFloat(document.getElementById('pointSize')?.value || 1);
    vp.raycaster.setFromCamera(vp.mouse, vp.camera);
    vp.raycaster.params.Points.threshold = pointSize * 5;

    try {
        const intersects = vp.raycaster.intersectObject(vp.pointCloud, false);
        if (intersects.length > 0) {
            const pointIndex = intersects[0].index;
            if (pointIndex !== undefined && pointIndex < vp.renderedIndicesMap.length) {
                const dataIdx = vp.renderedIndicesMap[pointIndex];
                if (dataIdx !== undefined && dataIdx < allData.length) {
                    const point = allData[dataIdx];
                    const pos = new THREE.Vector3(point.x, point.y, point.z);
                    showHighlight(pos, point, event.clientX, event.clientY, vp);
                    return;
                }
            }
        }
        hideHighlight();
    } catch (error) {
        hideHighlight();
    }
}

function showHighlight(position, point, mouseX, mouseY, vp) {
    if (!vp.highlightSphere || !tooltip) return;
    const pointSize = parseFloat(document.getElementById('pointSize')?.value || 1);
    const s = pointSize * 1.5;
    vp.highlightSphere.scale.set(s, s, s);
    vp.highlightSphere.position.copy(position);
    vp.highlightSphere.visible = true;

    const rows = [];
    rows.push(`<div><strong>${selectedCoordX}:</strong> ${Number(point.x).toFixed(2)}</div>`);
    rows.push(`<div><strong>${selectedCoordY}:</strong> ${Number(point.y).toFixed(2)}</div>`);
    rows.push(`<div><strong>${selectedCoordZ}:</strong> ${Number(point.z).toFixed(2)}</div>`);

    if (point.Pseudotime != null && point.Pseudotime !== '') {
        const pv = typeof point.Pseudotime === 'number'
            ? Number(point.Pseudotime).toFixed(4)
            : String(point.Pseudotime);
        rows.push(`<div><strong>Pseudotime:</strong> ${pv}</div>`);
    }

    // Show all currently loaded non-internal attributes for this point.
    const extraKeys = Object.keys(point)
        .filter((k) => !['x', 'y', 'z', 'Pseudotime'].includes(k) && !k.startsWith('_raw_'))
        .sort((a, b) => a.localeCompare(b));

    for (const key of extraKeys) {
        const value = point[key];
        if (value === undefined || value === null || value === '') continue;
        const displayName = getAttributeDisplayName(key);
        const displayValue = typeof value === 'number' ? value.toFixed(4) : String(value);
        rows.push(`<div><strong>${displayName}:</strong> ${displayValue}</div>`);
    }

    let html = '<div class="point-tooltip-title">Point Information</div>';
    html += rows.join('');
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    const off = 15;
    tooltip.style.left = (mouseX + off) + 'px';
    tooltip.style.top = (mouseY + off) + 'px';
}

function hideHighlight() {
    for (const key of ['wt', 'ko']) {
        if (viewports[key].highlightSphere) viewports[key].highlightSphere.visible = false;
    }
    if (tooltip) tooltip.style.display = 'none';
}

// Helper function to load a single Parquet file with progress tracking
async function loadSingleParquetFile(url, fileIndex, progressCallback) {
    const response = await fetch(url);
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} for ${url}`);
    }
    
    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    
    // Read with progress tracking
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        chunks.push(value);
        loaded += value.length;
        
        if (progressCallback) {
            progressCallback(fileIndex, loaded, total);
        }
    }
    
    // Combine chunks into single ArrayBuffer
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
    }
    
    return { buffer: buffer.buffer, size: loaded };
}

// Download all Parquet files in parallel (without parsing)
async function downloadAllParquetFiles(urls, onProgress) {
    // Track progress for each file
    const fileProgress = urls.map(() => ({ loaded: 0, total: 0 }));
    
    const updateProgress = (fileIndex, loaded, total) => {
        fileProgress[fileIndex] = { loaded, total };
        
        // Calculate total progress
        const totalLoaded = fileProgress.reduce((sum, p) => sum + p.loaded, 0);
        const totalSize = fileProgress.reduce((sum, p) => sum + p.total, 0);
        
        if (onProgress && totalSize > 0) {
            onProgress(totalLoaded, totalSize, fileProgress);
        }
    };
    
    console.log(`[Parquet] Starting parallel download of ${urls.length} files...`);
    const startTime = Date.now();
    
    // Download all files in parallel
    const downloadResults = await Promise.all(
        urls.map((url, index) => loadSingleParquetFile(url, index, updateProgress))
    );
    
    const totalSize = downloadResults.reduce((sum, r) => sum + r.size, 0);
    const elapsed = (Date.now() - startTime) / 1000;
    console.log(`[Parquet] Downloaded ${(totalSize / 1e6).toFixed(1)} MB in ${elapsed.toFixed(1)}s (${(totalSize / 1e6 / elapsed).toFixed(1)} MB/s)`);
    
    // Return buffers for later use
    return downloadResults.map(r => r.buffer);
}

// Parse specific columns from parquet buffers
async function parseParquetColumns(buffers, columns, onProgress) {
    console.log(`[Parquet] Parsing columns: ${columns.join(', ')}`);
    const startTime = Date.now();
    
    const allDataArrays = [];
    
    for (let i = 0; i < buffers.length; i++) {
        const buffer = buffers[i];
        
        await parquetRead({
            file: buffer,
            compressors,
            columns,
            onComplete: (data) => {
                if (onProgress) {
                    onProgress(i + 1, buffers.length);
                }
                allDataArrays.push(data);
            }
        });
    }
    
    // Concatenate all arrays
    const totalRows = allDataArrays.reduce((sum, arr) => sum + arr.length, 0);
    const elapsed = (Date.now() - startTime) / 1000;
    console.log(`[Parquet] Parsed ${totalRows.toLocaleString()} rows in ${elapsed.toFixed(1)}s`);
    
    // Flatten by iterating (avoids stack overflow)
    const allRows = new Array(totalRows);
    let idx = 0;
    for (const arr of allDataArrays) {
        for (let i = 0; i < arr.length; i++) {
            allRows[idx++] = arr[i];
        }
    }
    
    return allRows;
}

// Lazy load a column and merge into allData
async function lazyLoadColumn(columnName) {
    if (loadedColumns.has(columnName) || isLoadingColumn) {
        return;
    }
    
    console.log(`[Lazy Load] Loading column: ${columnName}`);
    isLoadingColumn = true;
    
    const loadingEl = document.getElementById('loading');
    const loadingText = loadingEl?.querySelector('.loading-text');
    if (loadingEl && loadingText) {
        loadingEl.style.display = 'flex';
        loadingText.textContent = `Loading ${columnName} data...`;
    }
    
    try {
        const startTime = Date.now();
        
        // Parse just this column from all buffers
        const columnData = await parseParquetColumns(parquetBuffers, [columnName], (current, total) => {
            if (loadingText) {
                loadingText.textContent = `Loading ${columnName}: file ${current}/${total}`;
            }
        });
        
        // Merge into allData
        // Note: hyparquet returns arrays, so we access index 0 since we only requested one column
        for (let i = 0; i < allData.length && i < columnData.length; i++) {
            const value = columnData[i][0];
            allData[i][columnName] = value;
            
            // Track unique values for categorical columns
            if (column_names_categorical.includes(columnName) && value != null) {
                const strValue = String(value);
                allData[i][columnName] = strValue;
                if (!attributeValues[columnName]) {
                    attributeValues[columnName] = new Set();
                }
                attributeValues[columnName].add(strValue);
            }
            
            // Track ranges for continuous columns
            if (column_names_continuous.includes(columnName) && value != null) {
                if (!continuousRanges[columnName]) {
                    continuousRanges[columnName] = { min: Infinity, max: -Infinity };
                }
                continuousRanges[columnName].min = Math.min(continuousRanges[columnName].min, value);
                continuousRanges[columnName].max = Math.max(continuousRanges[columnName].max, value);
            }
        }
        
        loadedColumns.add(columnName);
        
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`[Lazy Load] Loaded ${columnName} in ${elapsed.toFixed(1)}s`);
        
    } finally {
        isLoadingColumn = false;
        if (loadingEl) {
            loadingEl.style.display = 'none';
        }
    }
}

// Ensure a column is loaded (lazy load if needed)
async function ensureColumnLoaded(columnName) {
    if (!loadedColumns.has(columnName)) {
        await lazyLoadColumn(columnName);
    }
}

// Load additional shards to increase sample rate
async function loadMoreShards(targetShardCount) {
    if (isLoadingShards || targetShardCount <= loadedShardCount) {
        return;
    }
    
    // Cap at max shards
    targetShardCount = Math.min(targetShardCount, PARQUET_SHARDS.length);
    
    const shardsToLoad = targetShardCount - loadedShardCount;
    if (shardsToLoad <= 0) return;
    
    console.log(`[Shards] Loading ${shardsToLoad} more shards (${loadedShardCount + 1} to ${targetShardCount})...`);
    isLoadingShards = true;
    
    const loadingEl = document.getElementById('loading');
    const loadingText = loadingEl?.querySelector('.loading-text');
    if (loadingEl && loadingText) {
        loadingEl.style.display = 'flex';
    }
    
    try {
        const startTime = Date.now();
        
        // Get the shard URLs to load
        const shardUrls = PARQUET_SHARDS.slice(loadedShardCount, targetShardCount);
        
        // Download shards in parallel
        const newBuffers = await downloadAllParquetFiles(shardUrls, (loaded, total, fileProgress) => {
            if (loadingText) {
                const percent = Math.round((loaded / total) * 100);
                const loadedMB = (loaded / 1e6).toFixed(1);
                const totalMB = (total / 1e6).toFixed(1);
                loadingText.textContent = `Loading more data: ${loadedMB} / ${totalMB} MB (${percent}%)`;
            }
        });
        
        // Add to our buffer collection
        parquetBuffers.push(...newBuffers);
        
        // Parse the new shards with the columns we've already loaded
        const columnsToLoad = Array.from(loadedColumns);
        
        if (loadingText) {
            loadingText.textContent = 'Parsing new data...';
        }
        
        const newRows = await parseParquetColumns(newBuffers, columnsToLoad, (current, total) => {
            if (loadingText) {
                loadingText.textContent = `Parsing shard ${loadedShardCount + current}/${targetShardCount}...`;
            }
        });
        
        // Process new rows and add to allData
        if (loadingText) {
            loadingText.textContent = 'Processing new points...';
        }
        
        const defaultColorBy = column_names_categorical[0];
        
        // Create column index map (hyparquet returns arrays, not objects)
        const colIdx = {};
        columnsToLoad.forEach((col, idx) => {
            colIdx[col] = idx;
        });
        
        for (const row of newRows) {
            const point = {};
            
            // Store all raw coordinate column values
            for (const col of COORD_COLUMN_OPTIONS) {
                if (colIdx[col] !== undefined) {
                    point['_raw_' + col] = row[colIdx[col]] ?? 0;
                }
            }
            
            // Add loaded columns by index (before setting display coords so Group is known)
            for (const col of loadedColumns) {
                const idx = colIdx[col];
                if (idx === undefined) continue;
                
                if (col === 'Pseudotime') {
                    const pt = row[idx];
                    point[col] = pt;
                    if (pt != null && typeof pt === 'number') {
                        continuousRanges.Pseudotime.min = Math.min(continuousRanges.Pseudotime.min, pt);
                        continuousRanges.Pseudotime.max = Math.max(continuousRanges.Pseudotime.max, pt);
                    }
                } else if (column_names_categorical.includes(col)) {
                    const value = row[idx];
                    point[col] = value != null ? String(value) : '';
                    if (value && attributeValues[col]) {
                        attributeValues[col].add(point[col]);
                    }
                }
            }
            
            // Set display coordinates (after Group is known for KO offset)
            point.x = point['_raw_' + selectedCoordX] ?? 0;
            point.y = point['_raw_' + selectedCoordY] ?? 0;
            point.z = point['_raw_' + selectedCoordZ] ?? 0;
            if (point.Group === 'KO' && selectedCoordY === 'transformedZ') {
                point.y += KO_TRANSFORMED_Y_OFFSET;
            }
            
            allData.push(point);
        }
        
        // Update visible indices to include new data
        const oldLength = visibleIndices.length;
        const newVisibleIndices = new Uint32Array(allData.length);
        newVisibleIndices.set(visibleIndices);
        for (let i = oldLength; i < allData.length; i++) {
            newVisibleIndices[i] = i;
        }
        visibleIndices = newVisibleIndices;
        
        loadedShardCount = targetShardCount;
        
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`[Shards] Loaded ${shardsToLoad} shards in ${elapsed.toFixed(1)}s. Total points: ${allData.length.toLocaleString()}`);
        
        // Update UI
        document.getElementById('pointCount').textContent = `Total points: ${allData.length.toLocaleString()} (${loadedShardCount * 10}% loaded)`;
        
        // Rebuild TimeRank sliders with new data and re-filter
        const prevRanks = {};
        for (const vpKey of ['wt', 'ko']) {
            prevRanks[vpKey] = autoShiftTimeRankValues.length > 0
                ? autoShiftTimeRankValues[autoShiftCurrentIndex[vpKey]]
                : null;
        }
        buildTimeRankValues();
        for (const vpKey of ['wt', 'ko']) {
            const suffix = vpKey === 'wt' ? 'WT' : 'KO';
            const slider = document.getElementById('timerankSlider' + suffix);
            if (slider && autoShiftTimeRankValues.length > 0) {
                slider.max = autoShiftTimeRankValues.length - 1;
                if (prevRanks[vpKey] !== null) {
                    const newIdx = autoShiftTimeRankValues.indexOf(prevRanks[vpKey]);
                    autoShiftCurrentIndex[vpKey] = newIdx >= 0 ? newIdx : 0;
                }
                slider.value = autoShiftCurrentIndex[vpKey];
            }
        }
        syncAutoShiftUI();
        updateFilter();
        
    } finally {
        isLoadingShards = false;
        if (loadingEl) {
            loadingEl.style.display = 'none';
        }
    }
}

// Get required shard count for a given sample rate percentage
function getShardsForSampleRate(sampleRatePercent) {
    // Each shard is 10% of data
    // sampleRate 1-10 = 1 shard, 11-20 = 2 shards, etc.
    return Math.min(Math.ceil(sampleRatePercent / 10), PARQUET_SHARDS.length);
}

// Load and parse data from Parquet file
async function loadData() {
    const loadingEl = document.getElementById('loading');
    let loadingText = loadingEl.querySelector('.loading-text');
    
    // If loading-text doesn't exist, create the structure
    if (!loadingText) {
        loadingEl.innerHTML = '<div class="loading-spinner"></div><div class="loading-text">Loading data... This may take a moment.</div>';
        loadingText = loadingEl.querySelector('.loading-text');
    }
    
    // Ensure loading is visible
    loadingEl.style.display = 'flex';
    loadingText.textContent = 'Initializing...';
    
    try {
        const loadStartTime = Date.now();
        
        // Initialize data structures
        allData = [];
        
        // Initialize attribute tracking
        column_names_categorical.forEach(col => {
            attributeValues[col] = new Set();
        });
        column_names_continuous.forEach(col => {
            continuousRanges[col] = { min: Infinity, max: -Infinity };
        });
        
        // Download only the first shard initially (10% sample)
        // More shards will be loaded on-demand when user increases sample rate
        const initialShards = [PARQUET_SHARDS[0]];
        
        parquetBuffers = await downloadAllParquetFiles(initialShards, (loaded, total, fileProgress) => {
            const percent = Math.round((loaded / total) * 100);
            const loadedMB = (loaded / 1e6).toFixed(1);
            const totalMB = (total / 1e6).toFixed(1);
            loadingText.textContent = `Downloading: ${loadedMB} / ${totalMB} MB (${percent}%)`;
        });
        
        loadedShardCount = 1;
        
        // Define essential columns for initial load:
        // - All coordinate columns (for coordinate system switching)
        // - Default colorBy column (first categorical)
        // - Pseudotime (continuous attribute)
        const defaultColorBy = column_names_categorical[0]; // 'Structure'
        const essentialColumns = [
            ...COORD_COLUMN_OPTIONS,
            'Pseudotime',
            defaultColorBy,
            'Group',
            'HF',
            'Sample',
            'Gene'
        ];
        // De-duplicate in case defaultColorBy is already one of the above
        const essentialColumnsUnique = [...new Set(essentialColumns)];
        
        loadingText.textContent = 'Parsing essential columns...';
        await new Promise(resolve => setTimeout(resolve, 0));
        
        // Parse only essential columns
        const rows = await parseParquetColumns(parquetBuffers, essentialColumnsUnique, (current, total) => {
            loadingText.textContent = `Parsing file ${current}/${total}...`;
        });
        
        // Mark these columns as loaded
        essentialColumnsUnique.forEach(col => loadedColumns.add(col));
        
        // Create column index map (hyparquet returns arrays, not objects)
        const colIdx = {};
        essentialColumnsUnique.forEach((col, idx) => {
            colIdx[col] = idx;
        });
        
        const numRows = rows.length;
        console.log(`[Parquet] Processing ${numRows.toLocaleString()} rows`);
        
        loadingText.textContent = `Processing ${numRows.toLocaleString()} points...`;
        await new Promise(resolve => setTimeout(resolve, 0));
        
        // Process rows in chunks to keep UI responsive
        const CHUNK_SIZE = 500000;
        
        for (let startIdx = 0; startIdx < numRows; startIdx += CHUNK_SIZE) {
            const endIdx = Math.min(startIdx + CHUNK_SIZE, numRows);
            
            for (let i = startIdx; i < endIdx; i++) {
                const row = rows[i];
                
                const point = {};
                
                // Store all raw coordinate column values for axis switching
                for (const col of COORD_COLUMN_OPTIONS) {
                    if (colIdx[col] !== undefined) {
                        point['_raw_' + col] = row[colIdx[col]] ?? 0;
                    }
                }
                
                // Add the default colorBy attribute (already loaded)
                const rowDefaultColorBy = row[colIdx[defaultColorBy]];
                point[defaultColorBy] = rowDefaultColorBy != null ? String(rowDefaultColorBy) : '';
                if (rowDefaultColorBy) {
                    if (!attributeValues[defaultColorBy]) attributeValues[defaultColorBy] = new Set();
                    attributeValues[defaultColorBy].add(point[defaultColorBy]);
                }
                
                // Pseudotime (continuous attribute)
                if (colIdx.Pseudotime !== undefined) {
                    const pt = row[colIdx.Pseudotime];
                    point.Pseudotime = pt;
                    if (pt != null && typeof pt === 'number') {
                        continuousRanges.Pseudotime.min = Math.min(continuousRanges.Pseudotime.min, pt);
                        continuousRanges.Pseudotime.max = Math.max(continuousRanges.Pseudotime.max, pt);
                    }
                }
                
                // Eagerly load key categorical fields used in overlays/hover
                for (const catCol of ['Group', 'HF', 'Sample', 'Gene']) {
                    if (colIdx[catCol] !== undefined) {
                        const val = row[colIdx[catCol]];
                        point[catCol] = val != null ? String(val) : '';
                        if (val) {
                            if (!attributeValues[catCol]) attributeValues[catCol] = new Set();
                            attributeValues[catCol].add(point[catCol]);
                        }
                    }
                }
                
                // Set display coordinates (after Group is known for KO offset)
                point.x = point['_raw_' + selectedCoordX] ?? 0;
                point.y = point['_raw_' + selectedCoordY] ?? 0;
                point.z = point['_raw_' + selectedCoordZ] ?? 0;
                if (point.Group === 'KO' && selectedCoordY === 'transformedZ') {
                    point.y += KO_TRANSFORMED_Y_OFFSET;
                }
                
                allData.push(point);
            }
            
            // Update progress
            const percent = Math.round((endIdx / numRows) * 100);
            loadingText.textContent = `Processing: ${endIdx.toLocaleString()} / ${numRows.toLocaleString()} points (${percent}%)`;
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        const loadTime = ((Date.now() - loadStartTime) / 1000).toFixed(1);
        console.log(`[Parquet] Loaded ${allData.length.toLocaleString()} points in ${loadTime}s`);
        
        // Log sample points
        if (allData.length > 0) {
            for (let i = 0; i < Math.min(5, allData.length); i++) {
                const p = allData[i];
                console.log(`Sample point ${i + 1}: x=${p.x.toFixed(2)}, y=${p.y.toFixed(2)}, z=${p.z.toFixed(2)}`);
            }
        }
        
        loadingText.textContent = `Loaded ${allData.length.toLocaleString()} points. Initializing visualization...`;
        document.getElementById('pointCount').textContent = `Total points: ${allData.length.toLocaleString()} (${loadedShardCount * 10}% loaded)`;
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Calculate and log coordinate ranges
        if (allData.length > 0) {
            let xMin = Infinity, xMax = -Infinity;
            let yMin = Infinity, yMax = -Infinity;
            let zMin = Infinity, zMax = -Infinity;
            
            for (let i = 0; i < Math.min(10000, allData.length); i++) {
                const p = allData[i];
                xMin = Math.min(xMin, p.x);
                xMax = Math.max(xMax, p.x);
                yMin = Math.min(yMin, p.y);
                yMax = Math.max(yMax, p.y);
                zMin = Math.min(zMin, p.z);
                zMax = Math.max(zMax, p.z);
            }
            
            console.log(`Coordinate ranges (sampled from first ${Math.min(10000, allData.length)} points):`);
            console.log(`  ${selectedCoordX}: [${xMin.toFixed(2)}, ${xMax.toFixed(2)}] (span: ${(xMax - xMin).toFixed(2)})`);
            console.log(`  ${selectedCoordY}: [${yMin.toFixed(2)}, ${yMax.toFixed(2)}] (span: ${(yMax - yMin).toFixed(2)})`);
            console.log(`  ${selectedCoordZ}: [${zMin.toFixed(2)}, ${zMax.toFixed(2)}] (span: ${(zMax - zMin).toFixed(2)})`);
        }
        
        // Initialize visible indices
        visibleIndices = new Uint32Array(allData.length);
        for (let i = 0; i < allData.length; i++) {
            visibleIndices[i] = i;
        }
        
        // Populate coordinate axis dropdowns
        populateCoordDropdowns();
        
        // Populate colorBy dropdown
        const colorBySelect = document.getElementById('colorBy');
        colorBySelect.innerHTML = '';
        const allAttributes = [...UI_CATEGORICAL_OPTIONS];
        allAttributes.forEach(attr => {
            const option = document.createElement('option');
            option.value = attr;
            option.textContent = getAttributeDisplayName(attr);
            colorBySelect.appendChild(option);
        });
        if (allAttributes.length > 0) {
            colorBySelect.value = allAttributes[0];
        }
        
        // Build filter UI
        renderFilters();
        
        // Create initial visualization
        loadingText.textContent = 'Rendering visualization...';
        
        // Initialize TimeRank slider and filter to first hair follicle
        initTimerankSlider();
        
        // Initialize legend
        updateLegend();
        
        // Set up event listeners (only once)
        if (!eventListenersInitialized) {
            setupEventListeners();
            eventListenersInitialized = true;
            animate();
        }
        
        // Hide loading message
        setTimeout(() => {
            loadingEl.style.display = 'none';
        }, 500);
        
    } catch (error) {
        console.error('Error loading data:', error);
        loadingText.textContent = `Error loading data: ${error.message}`;
        loadingEl.style.background = 'rgba(231, 76, 60, 0.9)';
        loadingEl.style.borderColor = 'rgba(192, 57, 43, 0.5)';
    }
}

// Change color for a specific entity
function changeEntityColor(attribute, value, colorKey, colorDivElement) {
    // Get current color
    const currentColor = colorMap.get(colorKey) || getColorForValue(value, attribute);
    const currentHex = '#' + 
        Math.round(currentColor.r * 255).toString(16).padStart(2, '0') +
        Math.round(currentColor.g * 255).toString(16).padStart(2, '0') +
        Math.round(currentColor.b * 255).toString(16).padStart(2, '0');
    
    // Create a color input
    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = currentHex;
    colorInput.style.position = 'absolute';
    colorInput.style.opacity = '0';
    colorInput.style.width = '0';
    colorInput.style.height = '0';
    colorInput.style.pointerEvents = 'none';
    
    // Add to body temporarily
    document.body.appendChild(colorInput);
    
    // Trigger color picker
    colorInput.click();
    
    // Listen for change
    colorInput.addEventListener('change', (event) => {
        const newHex = event.target.value;
        
        // Parse hex color to THREE.Color
        const r = parseInt(newHex.slice(1, 3), 16) / 255;
        const g = parseInt(newHex.slice(3, 5), 16) / 255;
        const b = parseInt(newHex.slice(5, 7), 16) / 255;
        
        const newColor = new THREE.Color(r, g, b);
        
        // Update color map
        colorMap.set(colorKey, newColor);
        
        // Update the color div in legend
        colorDivElement.style.backgroundColor = newHex;
        
        createPointCloud();
        
        // Clean up
        document.body.removeChild(colorInput);
        
        console.log(`[Color Change] Changed color for ${attribute}:${value} to ${newHex}`);
    });
    
    // Also handle if user cancels (click outside)
    colorInput.addEventListener('blur', () => {
        setTimeout(() => {
            if (document.body.contains(colorInput)) {
                document.body.removeChild(colorInput);
            }
        }, 100);
    });
}

// Generate color for a value
function getColorForValue(value, attribute) {
    if (value === null || value === undefined || value === '') return new THREE.Color(0x888888);
    
    // For continuous variables, use a gradient based on the value
    if (column_names_continuous.includes(attribute) && typeof value === 'number') {
        const range = continuousRanges[attribute];
        if (range && range.max > range.min) {
            const normalized = (value - range.min) / (range.max - range.min);
            const color = new THREE.Color();
            // Use a color gradient from blue to red
            color.setHSL((1 - normalized) * 0.7, 0.8, 0.5);
            return color;
        }
    }
    
    const key = `${attribute}:${value}`;
    
    if (!colorMap.has(key)) {
        // Generate a color based on hash of the value
        const strValue = String(value);
        let hash = 0;
        for (let i = 0; i < strValue.length; i++) {
            hash = strValue.charCodeAt(i) + ((hash << 5) - hash);
        }
        
        const hue = (hash % 360 + 360) % 360;
        const saturation = 70 + (hash % 20);
        const lightness = 50 + (hash % 20);
        
        const color = new THREE.Color();
        color.setHSL(hue / 360, saturation / 100, lightness / 100);
        colorMap.set(key, color);
    }
    
    return colorMap.get(key);
}

// Randomize colors for categorical variables
function randomizeColors() {
    const colorBy = document.getElementById('colorBy').value;
    const isContinuous = column_names_continuous.includes(colorBy);
    
    if (isContinuous) {
        console.warn('[randomizeColors] Cannot randomize colors for continuous variables');
        return;
    }
    
    // Use pre-computed unique values instead of iterating all points
    const allValues = attributeValues[colorBy] || new Set();
    const sortedValues = Array.from(allValues).sort();
    const numValues = sortedValues.length;
    
    if (numValues === 0) {
        console.warn('[randomizeColors] No values to colorize');
        return;
    }
    
    // Generate rainbow colors evenly distributed across the hue spectrum
    // Create a shuffled array of hues
    const hues = [];
    for (let i = 0; i < numValues; i++) {
        hues.push((i / numValues) * 360);
    }
    
    // Shuffle the hues array
    for (let i = hues.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [hues[i], hues[j]] = [hues[j], hues[i]];
    }
    
    // Assign colors to values
    sortedValues.forEach((value, index) => {
        const colorKey = `${colorBy}:${value}`;
        const color = new THREE.Color();
        // Use full saturation and medium lightness for vibrant colors
        color.setHSL(hues[index] / 360, 0.8, 0.5);
        colorMap.set(colorKey, color);
    });
    
    console.log(`[randomizeColors] Randomized colors for ${numValues} values`);
    
    createPointCloud();
    
    // Update the legend to show new colors
    updateLegend();
}

// Update the color legend
function updateLegend() {
    const legendDiv = document.getElementById('legend');
    if (!legendDiv) return;
    
    const colorBy = document.getElementById('colorBy').value;
    const isContinuous = column_names_continuous.includes(colorBy);
    
    // Use pre-computed attribute values instead of iterating all points
    // This avoids O(14M) iteration on every legend update
    let visibleValues;
    if (isContinuous) {
        // For continuous values, use the pre-computed range
        const range = continuousRanges[colorBy];
        visibleValues = range ? new Set([range.min, range.max]) : new Set();
    } else {
        // For categorical values, use the pre-computed unique values
        visibleValues = attributeValues[colorBy] || new Set();
    }
    
    legendDiv.innerHTML = '';
    
    // Show/hide randomize colors button based on variable type
    const randomizeButton = document.getElementById('randomizeColors');
    if (randomizeButton) {
        randomizeButton.style.display = isContinuous ? 'none' : 'block';
    }
    
    if (isContinuous) {
        // Show gradient for continuous values
        if (visibleValues.size > 0) {
            const continuousValues = Array.from(visibleValues).map(v => parseFloat(v)).filter(v => !isNaN(v));
            if (continuousValues.length > 0) {
                // Use reduce to avoid stack overflow with large arrays
                const minVal = continuousValues.reduce((min, val) => val < min ? val : min, continuousValues[0]);
                const maxVal = continuousValues.reduce((max, val) => val > max ? val : max, continuousValues[0]);
                
                // Create gradient canvas
                const canvas = document.createElement('canvas');
                canvas.width = 200;
                canvas.height = 30;
                canvas.className = 'legend-gradient';
                const ctx = canvas.getContext('2d');
                
                const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
                for (let i = 0; i <= 100; i++) {
                    const normalized = i / 100;
                    const value = minVal + (maxVal - minVal) * normalized;
                    const color = getColorForValue(value, colorBy);
                    const stop = i / 100;
                    gradient.addColorStop(stop, `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`);
                }
                
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                legendDiv.appendChild(canvas);
                
                const labelsDiv = document.createElement('div');
                labelsDiv.className = 'legend-gradient-labels';
                labelsDiv.innerHTML = `<span>${minVal.toFixed(4)}</span><span>${maxVal.toFixed(4)}</span>`;
                legendDiv.appendChild(labelsDiv);
            } else {
                legendDiv.innerHTML = `<div class="legend-label">No ${getAttributeDisplayName(colorBy)} data</div>`;
            }
        } else {
            legendDiv.innerHTML = '<div class="legend-label">No visible data</div>';
        }
    } else {
        // Show categorical legend
        const sortedValues = Array.from(visibleValues).sort();
        
        if (sortedValues.length === 0) {
            legendDiv.innerHTML = '<div class="legend-label">No visible data</div>';
            return;
        }
        
        // Limit to first 100 items for performance
        const displayValues = sortedValues.slice(0, 100);
        const remainingCount = sortedValues.length - displayValues.length;
        
        displayValues.forEach(value => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'legend-item';
            itemDiv.style.cursor = 'pointer';
            itemDiv.title = 'Click to change color';
            
            const colorDiv = document.createElement('div');
            colorDiv.className = 'legend-color';
            const color = getColorForValue(value, colorBy);
            const colorKey = `${colorBy}:${value}`;
            colorDiv.style.backgroundColor = `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'legend-label';
            labelDiv.textContent = value || '(empty)';
            
            // Add click handler to change color
            itemDiv.addEventListener('click', (event) => {
                event.stopPropagation();
                changeEntityColor(colorBy, value, colorKey, colorDiv);
            });
            
            // Add hover effect
            itemDiv.addEventListener('mouseenter', () => {
                itemDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                itemDiv.style.borderRadius = '4px';
            });
            
            itemDiv.addEventListener('mouseleave', () => {
                itemDiv.style.backgroundColor = 'transparent';
            });
            
            itemDiv.appendChild(colorDiv);
            itemDiv.appendChild(labelDiv);
            legendDiv.appendChild(itemDiv);
        });
        
        if (remainingCount > 0) {
            const moreDiv = document.createElement('div');
            moreDiv.className = 'legend-label';
            moreDiv.style.fontStyle = 'italic';
            moreDiv.style.color = '#95a5a6';
            moreDiv.textContent = `... and ${remainingCount} more`;
            legendDiv.appendChild(moreDiv);
        }
    }
}

// Set camera to x-y plane view
// Set both viewports' cameras to x-y plane view based on geometry bounds
function setCameraToXYPlaneView(geometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = box.getSize(new THREE.Vector3());
    const xyExtent = Math.max(size.x, size.y);

    for (const key of ['wt', 'ko']) {
        const vp = viewports[key];
        const fovRad = vp.camera.fov * (Math.PI / 180);
        const halfHeight = xyExtent / 2;
        const distance = halfHeight / Math.tan(fovRad / 2);
        const cameraHeight = Math.max(distance * 1.1, size.z * 1.5, 100);

        const pos = new THREE.Vector3(center.x, center.y, center.z + cameraHeight);
        vp.camera.position.copy(pos);
        vp.camera.lookAt(center);
        vp.controls.target.copy(center);
        vp.controls.update();
    }

    initialCameraState.position = new THREE.Vector3(center.x, center.y, center.z + Math.max(xyExtent * 1.1, size.z * 1.5, 100));
    initialCameraState.target = center.clone();
}

// Shared shader sources
const pointVertexShader = `
    attribute vec3 color;
    varying vec3 vColor;
    uniform float pointSize;
    void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = pointSize * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
    }
`;
const pointFragmentShader = `
    varying vec3 vColor;
    uniform float opacity;
    void main() {
        vec2 center = gl_PointCoord - vec2(0.5);
        float dist = length(center);
        if (dist > 0.5) discard;
        float alpha = opacity * (1.0 - smoothstep(0.45, 0.5, dist));
        gl_FragColor = vec4(vColor, alpha);
    }
`;

// Build a point cloud for a subset of indices and add it to a viewport
function buildPointCloudForViewport(vpKey, indices) {
    const vp = viewports[vpKey];
    if (vp.pointCloud) {
        vp.scene.remove(vp.pointCloud);
        vp.pointCloud.geometry.dispose();
        vp.pointCloud.material.dispose();
        vp.pointCloud = null;
    }
    if (vp.highlightSphere) vp.highlightSphere.visible = false;

    if (!indices || indices.length === 0) {
        vp.renderedIndicesMap = null;
        return;
    }

    const colorBy = document.getElementById('colorBy').value;
    const pointSize = parseFloat(document.getElementById('pointSize').value);
    const count = Math.min(indices.length, MAX_POINTS);
    const indicesToRender = indices.subarray ? indices.subarray(0, count) : indices.slice(0, count);

    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const point = allData[indicesToRender[i]];
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
        const c = getColorForValue(point[colorBy], colorBy);
        colors[i * 3] = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.ShaderMaterial({
        uniforms: { pointSize: { value: pointSize * 2 }, opacity: { value: 0.8 } },
        vertexShader: pointVertexShader,
        fragmentShader: pointFragmentShader,
        transparent: true,
        depthWrite: false,
    });

    vp.pointCloud = new THREE.Points(geometry, material);
    vp.scene.add(vp.pointCloud);
    vp.renderedIndicesMap = indicesToRender;

    return positions;
}

// Create point clouds for both viewports, splitting by Group
function createPointCloud() {
    if (!visibleIndices || visibleIndices.length === 0) {
        for (const key of ['wt', 'ko']) buildPointCloudForViewport(key, null);
        document.getElementById('visibleCount').textContent = 'Visible points: 0';
        updateViewportInfo();
        return;
    }

    // Split visible indices by Group
    const wtIndices = [];
    const koIndices = [];
    for (let i = 0; i < visibleIndices.length; i++) {
        const idx = visibleIndices[i];
        const group = allData[idx].Group;
        if (group === 'KO') {
            koIndices.push(idx);
        } else {
            wtIndices.push(idx);
        }
    }

    const wtArr = new Uint32Array(wtIndices);
    const koArr = new Uint32Array(koIndices);

    const wtPositions = buildPointCloudForViewport('wt', wtArr);
    buildPointCloudForViewport('ko', koArr);

    updateLegend();
    updateViewportInfo();

    const totalRendered = Math.min(wtIndices.length, MAX_POINTS) + Math.min(koIndices.length, MAX_POINTS);
    document.getElementById('visibleCount').textContent = `Rendering: ${totalRendered.toLocaleString()} points (WT: ${Math.min(wtIndices.length, MAX_POINTS).toLocaleString()}, KO: ${Math.min(koIndices.length, MAX_POINTS).toLocaleString()})`;

    if (!cameraInitialized && wtPositions && wtPositions.length > 0) {
        const tempGeometry = new THREE.BufferGeometry();
        tempGeometry.setAttribute('position', new THREE.BufferAttribute(wtPositions, 3));
        tempGeometry.computeBoundingBox();
        setCameraToXYPlaneView(tempGeometry);
        tempGeometry.dispose();
        cameraInitialized = true;
    }
}

// Update the info overlays in each viewport with HF, Sample, TimeRank, Pseudotime
function updateViewportInfo() {
    const wtInfoEl = document.getElementById('viewport-info-wt');
    const koInfoEl = document.getElementById('viewport-info-ko');
    if (!wtInfoEl || !koInfoEl) return;

    if (autoShiftTimeRankValues.length === 0) {
        wtInfoEl.innerHTML = '';
        koInfoEl.innerHTML = '';
        return;
    }

    for (const [group, el, vpKey] of [['WT', wtInfoEl, 'wt'], ['KO', koInfoEl, 'ko']]) {
        const targetRank = autoShiftTimeRankValues[autoShiftCurrentIndex[vpKey]];
        let hfVal = '—', sampleVal = '—', trVal = targetRank, pseudotimeVal = '—';
        if (visibleIndices) {
            for (let i = 0; i < visibleIndices.length; i++) {
                const p = allData[visibleIndices[i]];
                const pGroup = p.Group || '';
                const isMatch = (group === 'WT') ? (pGroup !== 'KO') : (pGroup === 'KO');
                if (isMatch && p._raw_TimeRank === targetRank) {
                    hfVal = p.HF || '—';
                    sampleVal = p.Sample || '—';
                    if (p.Pseudotime != null && p.Pseudotime !== '') {
                        pseudotimeVal = typeof p.Pseudotime === 'number'
                            ? Number(p.Pseudotime).toFixed(4)
                            : String(p.Pseudotime);
                    }
                    break;
                }
            }
        }
        el.innerHTML = `<div><strong>HF:</strong> ${hfVal}</div><div><strong>Sample:</strong> ${sampleVal}</div><div><strong>TimeRank:</strong> ${trVal}</div><div><strong>Pseudotime:</strong> ${pseudotimeVal}</div>`;
    }
}

// Create a filter UI element
function createFilterElement(filterId, attribute) {
    const filterDiv = document.createElement('div');
    filterDiv.className = 'filter-block';
    filterDiv.dataset.filterId = filterId;
    
    const availableAttributes = [...UI_CATEGORICAL_OPTIONS];
    const availableOptions = availableAttributes.map(attr => 
        `<option value="${attr}" ${attr === attribute ? 'selected' : ''}>${getAttributeDisplayName(attr)}</option>`
    ).join('');
    
    if (column_names_continuous.includes(attribute)) {
        // Continuous filter with sliders
        const range = continuousRanges[attribute];
        if (!range) {
            filterDiv.innerHTML = `<div class="filter-label">No range data for ${attribute}</div>`;
            return filterDiv;
        }
        
        // Get the filter object to check if it has existing range values
        const filter = activeFilters.find(f => f.id === filterId);
        const currentMin = filter && filter.range ? filter.range.min : range.min;
        const currentMax = filter && filter.range ? filter.range.max : range.max;
        
        const rangeSize = range.max - range.min;
        const stepSize = rangeSize > 0 ? Math.max(rangeSize / 1000, 0.0001) : 0.0001;
        
        filterDiv.innerHTML = `
            <div class="filter-header">
                <select class="filter-attribute" data-filter-id="${filterId}">
                    <option value="">Select attribute...</option>
                    ${availableOptions}
                </select>
                <button class="remove-filter" data-filter-id="${filterId}">×</button>
            </div>
            <div class="filter-content" data-filter-id="${filterId}">
                <label>${attribute} Range:</label>
                <div class="time-slider-wrapper">
                    <input type="range" class="time-min" data-filter-id="${filterId}" 
                           min="${range.min}" max="${range.max}" 
                           step="${stepSize}" value="${currentMin}">
                    <input type="range" class="time-max" data-filter-id="${filterId}" 
                           min="${range.min}" max="${range.max}" 
                           step="${stepSize}" value="${currentMax}">
                </div>
                <div class="time-values">
                    <span>Min: <span class="time-min-value">${currentMin.toFixed(4)}</span></span>
                    <span>Max: <span class="time-max-value">${currentMax.toFixed(4)}</span></span>
                </div>
            </div>
        `;
    } else if (attribute) {
        // Categorical filter with checkboxes
        const values = Array.from(attributeValues[attribute] || []).sort();
        const filter = activeFilters.find(f => f.id === filterId);
        const selectedValues = filter && filter.values ? filter.values : new Set(values);
        
        const checkboxes = values.map(value => {
            const checked = selectedValues.has(value) ? 'checked' : '';
            return `
            <div class="filter-checkbox-item">
                <input type="checkbox" class="filter-checkbox" data-filter-id="${filterId}" 
                       value="${value}" ${checked}>
                <label>${value || '(empty)'}</label>
            </div>
        `;
        }).join('');
        
        filterDiv.innerHTML = `
            <div class="filter-header">
                <select class="filter-attribute" data-filter-id="${filterId}">
                    <option value="">Select attribute...</option>
                    ${availableOptions}
                </select>
                <button class="remove-filter" data-filter-id="${filterId}">×</button>
            </div>
            <div class="filter-content" data-filter-id="${filterId}">
                <div class="filter-checkboxes-container">
                    ${checkboxes}
                </div>
                <div class="filter-buttons">
                    <button class="select-all-filter" data-filter-id="${filterId}">Select All</button>
                    <button class="deselect-all-filter" data-filter-id="${filterId}">Deselect All</button>
                </div>
            </div>
        `;
    } else {
        // Empty filter
        filterDiv.innerHTML = `
            <div class="filter-header">
                <select class="filter-attribute" data-filter-id="${filterId}">
                    <option value="">Select attribute...</option>
                    ${availableOptions}
                </select>
                <button class="remove-filter" data-filter-id="${filterId}">×</button>
            </div>
            <div class="filter-content" data-filter-id="${filterId}">
                <p style="color: #95a5a6; font-size: 0.85em;">Select an attribute to filter by</p>
            </div>
        `;
    }
    
    return filterDiv;
}

// Render all active filters
function renderFilters() {
    const container = document.getElementById('filtersContainer');
    container.innerHTML = '';
    
    if (activeFilters.length === 0) {
        return;
    }
    
    activeFilters.forEach((filter, index) => {
        const filterElement = createFilterElement(filter.id, filter.attribute);
        container.appendChild(filterElement);
    });
    
    // Attach event listeners
    attachFilterEventListeners();
}

// Attach event listeners to filter elements
function attachFilterEventListeners() {
    // Attribute change
    document.querySelectorAll('.filter-attribute').forEach(select => {
        // Remove existing listeners by cloning
        const newSelect = select.cloneNode(true);
        select.parentNode.replaceChild(newSelect, select);
        
        newSelect.addEventListener('change', async (e) => {
            const filterId = e.target.dataset.filterId;
            const attribute = e.target.value;
            
            // Lazy load the column if not already loaded
            if (attribute && !loadedColumns.has(attribute)) {
                await lazyLoadColumn(attribute);
            }
            
            const filter = activeFilters.find(f => f.id === filterId);
            if (filter) {
                filter.attribute = attribute;
                if (column_names_continuous.includes(attribute)) {
                    filter.type = 'continuous';
                    const range = continuousRanges[attribute];
                    filter.range = range ? { min: range.min, max: range.max } : { min: 0, max: 1 };
                    filter.values = null;
                } else if (attribute && column_names_categorical.includes(attribute)) {
                    filter.type = 'categorical';
                    const values = Array.from(attributeValues[attribute] || []);
                    filter.values = new Set(values);
                    filter.range = null;
                } else {
                    filter.type = null;
                    filter.values = null;
                    filter.range = null;
                }
                renderFilters();
                throttleFilterUpdate();
            }
        });
    });
    
    // Remove filter
    document.querySelectorAll('.remove-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const filterId = e.target.dataset.filterId;
            const filterToRemove = activeFilters.find(f => f.id === filterId);
            if (filterToRemove) {
                console.log('[Filter] Removing filter:', {
                    id: filterId,
                    attribute: filterToRemove.attribute,
                    type: filterToRemove.type
                });
            }
            activeFilters = activeFilters.filter(f => f.id !== filterId);
            console.log('[Filter] Active filters after removal:', activeFilters.length);
            renderFilters();
            throttleFilterUpdate();
        });
    });
    
    // Time sliders - need to attach fresh each time
    document.querySelectorAll('.time-min').forEach(slider => {
        // Clone to remove old listeners
        const newSlider = slider.cloneNode(true);
        slider.parentNode.replaceChild(newSlider, slider);
        
        newSlider.addEventListener('input', (e) => {
            const filterId = e.target.dataset.filterId;
            const minVal = parseFloat(e.target.value);
            const filter = activeFilters.find(f => f.id === filterId);
            
            if (filter && filter.range) {
                if (minVal > filter.range.max) {
                    filter.range.max = minVal;
                    const maxSlider = document.querySelector(`.time-max[data-filter-id="${filterId}"]`);
                    if (maxSlider) maxSlider.value = minVal;
                    const maxValueEl = document.querySelector(`.time-max-value[data-filter-id="${filterId}"]`);
                    if (maxValueEl) maxValueEl.textContent = minVal.toFixed(4);
                }
                filter.range.min = minVal;
                const minValueEl = document.querySelector(`.time-min-value[data-filter-id="${filterId}"]`);
                if (minValueEl) minValueEl.textContent = minVal.toFixed(4);
                throttleFilterUpdate();
            }
        });
    });
    
    document.querySelectorAll('.time-max').forEach(slider => {
        // Clone to remove old listeners
        const newSlider = slider.cloneNode(true);
        slider.parentNode.replaceChild(newSlider, slider);
        
        newSlider.addEventListener('input', (e) => {
            const filterId = e.target.dataset.filterId;
            const maxVal = parseFloat(e.target.value);
            const filter = activeFilters.find(f => f.id === filterId);
            
            if (filter && filter.range) {
                if (maxVal < filter.range.min) {
                    filter.range.min = maxVal;
                    const minSlider = document.querySelector(`.time-min[data-filter-id="${filterId}"]`);
                    if (minSlider) minSlider.value = maxVal;
                    const minValueEl = document.querySelector(`.time-min-value[data-filter-id="${filterId}"]`);
                    if (minValueEl) minValueEl.textContent = maxVal.toFixed(4);
                }
                filter.range.max = maxVal;
                const maxValueEl = document.querySelector(`.time-max-value[data-filter-id="${filterId}"]`);
                if (maxValueEl) maxValueEl.textContent = maxVal.toFixed(4);
                throttleFilterUpdate();
            }
        });
    });
    
    // Categorical checkboxes
    document.querySelectorAll('.filter-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const filterId = checkbox.dataset.filterId;
            const filter = activeFilters.find(f => f.id === filterId);
            
            if (filter && filter.values) {
                const allCheckboxes = document.querySelectorAll(`.filter-checkbox[data-filter-id="${filterId}"]`);
                filter.values.clear();
                allCheckboxes.forEach(cb => {
                    if (cb.checked) {
                        filter.values.add(cb.value);
                    }
                });
                throttleFilterUpdate();
            }
        });
    });
    
    // Select all / Deselect all buttons
    document.querySelectorAll('.select-all-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const filterId = e.target.dataset.filterId;
            document.querySelectorAll(`.filter-checkbox[data-filter-id="${filterId}"]`).forEach(cb => {
                cb.checked = true;
            });
            const filter = activeFilters.find(f => f.id === filterId);
            if (filter && filter.values) {
                const allCheckboxes = document.querySelectorAll(`.filter-checkbox[data-filter-id="${filterId}"]`);
                filter.values.clear();
                allCheckboxes.forEach(cb => filter.values.add(cb.value));
                throttleFilterUpdate();
            }
        });
    });
    
    document.querySelectorAll('.deselect-all-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const filterId = e.target.dataset.filterId;
            document.querySelectorAll(`.filter-checkbox[data-filter-id="${filterId}"]`).forEach(cb => {
                cb.checked = false;
            });
            const filter = activeFilters.find(f => f.id === filterId);
            if (filter && filter.values) {
                filter.values.clear();
                throttleFilterUpdate();
            }
        });
    });
}

// Throttle function for filter updates
let filterUpdateTimeout = null;
function throttleFilterUpdate() {
    if (filterUpdateTimeout) {
        clearTimeout(filterUpdateTimeout);
    }
    filterUpdateTimeout = setTimeout(() => {
        updateFilter();
    }, 100); // 100ms throttle
}

// Update filter - optimized for performance, applies all active filters
function updateFilter() {
    console.log('[Filter] Updating filters, active filters:', activeFilters.length);
    
    // Start with all indices
    let candidateIndices = [];
    for (let i = 0; i < allData.length; i++) {
        candidateIndices.push(i);
    }
    
    // Apply each user filter sequentially (AND logic)
    activeFilters.forEach((filter, index) => {
        if (!filter.attribute || !filter.type) {
            console.log(`[Filter] Skipping filter ${index + 1} (no attribute or type)`);
            return;
        }
        
        const filteredIndices = [];
        const beforeCount = candidateIndices.length;
        
        if (filter.type === 'continuous' && filter.range) {
            // Continuous range filter
            const minVal = filter.range.min;
            const maxVal = filter.range.max;
            
            for (let idx of candidateIndices) {
                const point = allData[idx];
                const value = point[filter.attribute];
                if (value !== null && value !== undefined && value >= minVal && value <= maxVal) {
                    filteredIndices.push(idx);
                }
            }
            console.log(`[Filter] Applied continuous filter on ${filter.attribute}: ${beforeCount} -> ${filteredIndices.length} points`);
        } else if (filter.type === 'categorical' && filter.values && filter.values.size > 0) {
            // Categorical filter
            for (let idx of candidateIndices) {
                const point = allData[idx];
                if (filter.values.has(point[filter.attribute] || '')) {
                    filteredIndices.push(idx);
                }
            }
            console.log(`[Filter] Applied categorical filter on ${filter.attribute}: ${beforeCount} -> ${filteredIndices.length} points`);
        } else {
            // Invalid filter, skip it (don't filter anything)
            console.log(`[Filter] Skipping invalid filter ${index + 1} on ${filter.attribute}`);
            return;
        }
        
        candidateIndices = filteredIndices;
    });
    
    // Convert to Uint32Array
    visibleIndices = new Uint32Array(candidateIndices);
    
    console.log(`[Filter] Final visible points: ${visibleIndices.length} out of ${allData.length} total`);
    
    // Filter each group to its own selected TimeRank value
    if (timerankFilterActive && autoShiftTimeRankValues.length > 0) {
        const wtRank = autoShiftTimeRankValues[autoShiftCurrentIndex.wt];
        const koRank = autoShiftTimeRankValues[autoShiftCurrentIndex.ko];
        const shifted = [];
        for (let i = 0; i < visibleIndices.length; i++) {
            const point = allData[visibleIndices[i]];
            const group = point.Group || '';
            const targetRank = (group === 'KO') ? koRank : wtRank;
            if (point._raw_TimeRank === targetRank) {
                shifted.push(visibleIndices[i]);
            }
        }
        visibleIndices = new Uint32Array(shifted);
    }

    createPointCloud();
}

// Build sorted unique TimeRank values from current data
function buildTimeRankValues() {
    const seen = new Set();
    for (let i = 0; i < allData.length; i++) {
        const v = allData[i]._raw_TimeRank;
        if (v !== undefined && v !== null) seen.add(v);
    }
    autoShiftTimeRankValues = Array.from(seen).sort((a, b) => a - b);
    console.log(`[AutoShift] Found ${autoShiftTimeRankValues.length} unique TimeRank values`);
}

// Update both TimeRank sliders and labels to reflect current indices
function syncAutoShiftUI() {
    for (const vpKey of ['wt', 'ko']) {
        const suffix = vpKey === 'wt' ? 'WT' : 'KO';
        const slider = document.getElementById('timerankSlider' + suffix);
        const statusEl = document.getElementById('timerankStatus' + suffix);
        if (slider) slider.value = autoShiftCurrentIndex[vpKey];
        if (statusEl && autoShiftTimeRankValues.length > 0) {
            const idx = autoShiftCurrentIndex[vpKey];
            statusEl.textContent = `${autoShiftTimeRankValues[idx]} (${idx + 1}/${autoShiftTimeRankValues.length})`;
        }
    }
    updateViewportInfo();
}

// Advance to the next TimeRank and re-filter
function autoShiftTick() {
    if (!autoShiftEnabled || autoShiftTimeRankValues.length === 0) return;

    autoShiftCurrentIndex.wt = (autoShiftCurrentIndex.wt + 1) % autoShiftTimeRankValues.length;
    autoShiftCurrentIndex.ko = (autoShiftCurrentIndex.ko + 1) % autoShiftTimeRankValues.length;
    syncAutoShiftUI();
    updateFilter();
}

// Initialize the TimeRank sliders after data is loaded
function initTimerankSlider() {
    buildTimeRankValues();
    if (autoShiftTimeRankValues.length === 0) {
        return;
    }
    const defaultIndex = autoShiftTimeRankValues.indexOf(DEFAULT_TIMERANK);
    const startIdx = defaultIndex >= 0 ? defaultIndex : 0;
    autoShiftCurrentIndex.wt = startIdx;
    autoShiftCurrentIndex.ko = startIdx;

    for (const suffix of ['WT', 'KO']) {
        const slider = document.getElementById('timerankSlider' + suffix);
        if (slider) {
            slider.min = 0;
            slider.max = autoShiftTimeRankValues.length - 1;
            slider.value = startIdx;
        }
    }
    syncAutoShiftUI();
    updateFilter();
}

// Start the auto-play animation
function startAutoShift() {
    if (autoShiftTimeRankValues.length === 0) {
        console.warn('[AutoShift] No TimeRank values found');
        return;
    }
    autoShiftEnabled = true;
    autoShiftIntervalId = setInterval(autoShiftTick, AUTO_SHIFT_INTERVAL_MS);
    console.log('[AutoShift] Started');
}

// Stop the auto-play animation (slider stays, filtering stays)
function stopAutoShift() {
    autoShiftEnabled = false;
    if (autoShiftIntervalId !== null) {
        clearInterval(autoShiftIntervalId);
        autoShiftIntervalId = null;
    }
    console.log('[AutoShift] Stopped');
}

// No-op: coordinate preset dropdown is static HTML
function populateCoordDropdowns() {}

// Remap coordinates from stored raw column values
function remapCoordinates() {
    console.log(`[Coordinates] Remapping to X=${selectedCoordX}, Y=${selectedCoordY}, Z=${selectedCoordZ}`);

    for (let i = 0; i < allData.length; i++) {
        const point = allData[i];
        point.x = point['_raw_' + selectedCoordX] ?? 0;
        point.y = point['_raw_' + selectedCoordY] ?? 0;
        point.z = point['_raw_' + selectedCoordZ] ?? 0;
        if (point.Group === 'KO' && selectedCoordY === 'transformedZ') {
            point.y += KO_TRANSFORMED_Y_OFFSET;
        }
    }

    cameraInitialized = false;

    colorMap.clear();

    // Re-apply existing filters with the new coordinates
    updateFilter();

    updateLegend();
}

// Setup event listeners
function setupEventListeners() {
    // Coordinate preset change handler
    document.getElementById('coordPreset').addEventListener('change', (e) => {
        const preset = COORD_PRESETS[e.target.value];
        if (!preset) return;
        selectedCoordX = preset.x;
        selectedCoordY = preset.y;
        selectedCoordZ = preset.z;
        remapCoordinates();
    });
    
    // Randomize colors button
    const randomizeButton = document.getElementById('randomizeColors');
    if (randomizeButton) {
        randomizeButton.addEventListener('click', () => {
            randomizeColors();
        });
    }
    
    document.getElementById('colorBy').addEventListener('change', async () => {
        const colorBy = document.getElementById('colorBy').value;
        
        // Lazy load the column if not already loaded
        if (!loadedColumns.has(colorBy)) {
            await lazyLoadColumn(colorBy);
        }
        
        createPointCloud();
    });
    
    // Add filter button
    document.getElementById('addFilter').addEventListener('click', () => {
        const filterId = 'filter_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const newFilter = {
            id: filterId,
            attribute: '',
            type: null,
            values: null,
            range: null
        };
        activeFilters.push(newFilter);
        console.log('[Filter] Created new filter:', {
            id: filterId,
            totalFilters: activeFilters.length
        });
        renderFilters();
    });
    
    document.getElementById('pointSize').addEventListener('input', (e) => {
        const newSize = parseFloat(e.target.value);
        document.getElementById('pointSizeValue').textContent = newSize.toFixed(1);
        for (const key of ['wt', 'ko']) {
            const pc = viewports[key].pointCloud;
            if (pc && pc.material && pc.material.uniforms) {
                pc.material.uniforms.pointSize.value = newSize * 2;
            }
        }
    });
    
    document.getElementById('sampleRate').addEventListener('input', async (e) => {
        const sampleRatePercent = parseInt(e.target.value);
        document.getElementById('sampleRateValue').textContent = sampleRatePercent + '%';
        
        // Calculate how many shards we need for this sample rate
        const requiredShards = getShardsForSampleRate(sampleRatePercent);
        
        // Load more shards if needed
        if (requiredShards > loadedShardCount) {
            await loadMoreShards(requiredShards);
        } else {
            // Just re-render with current data
            createPointCloud();
        }
    });
    
    
    document.getElementById('resetCamera').addEventListener('click', () => {
        const positions = [];
        if (visibleIndices && visibleIndices.length > 0) {
            const MAX_SAMPLE = 10000;
            const step = Math.max(1, Math.floor(visibleIndices.length / MAX_SAMPLE));
            for (let i = 0; i < visibleIndices.length; i += step) {
                const point = allData[visibleIndices[i]];
                positions.push(point.x, point.y, point.z);
            }
        }
        if (positions.length > 0) {
            const tempGeometry = new THREE.BufferGeometry();
            tempGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
            tempGeometry.computeBoundingBox();
            setCameraToXYPlaneView(tempGeometry);
            tempGeometry.dispose();
        } else if (initialCameraState.position && initialCameraState.target) {
            for (const key of ['wt', 'ko']) {
                const vp = viewports[key];
                vp.camera.position.copy(initialCameraState.position);
                vp.controls.target.copy(initialCameraState.target);
                vp.camera.lookAt(initialCameraState.target);
                vp.controls.update();
            }
        }
    });
    
    // Auto-rotate checkbox
    document.getElementById('autoRotate').addEventListener('change', (e) => {
        autoRotateEnabled = e.target.checked;
        console.log('[Camera] Auto-rotate:', autoRotateEnabled ? 'enabled' : 'disabled');
    });

    // TimeRank sliders – scrub to a specific hair follicle per viewport
    for (const [vpKey, suffix] of [['wt', 'WT'], ['ko', 'KO']]) {
        const slider = document.getElementById('timerankSlider' + suffix);

        slider.addEventListener('input', (e) => {
            const idx = parseInt(e.target.value);
            if (isNaN(idx) || idx < 0 || idx >= autoShiftTimeRankValues.length) return;

            const delta = idx - autoShiftCurrentIndex[vpKey];
            autoShiftCurrentIndex[vpKey] = idx;

            if (timerankCoupled) {
                const otherKey = vpKey === 'wt' ? 'ko' : 'wt';
                let otherIdx = autoShiftCurrentIndex[otherKey] + delta;
                otherIdx = Math.max(0, Math.min(autoShiftTimeRankValues.length - 1, otherIdx));
                autoShiftCurrentIndex[otherKey] = otherIdx;
            }

            syncAutoShiftUI();

            // Pause auto-play while dragging
            if (autoShiftIntervalId !== null) {
                clearInterval(autoShiftIntervalId);
                autoShiftIntervalId = null;
            }

            updateFilter();
        });

        slider.addEventListener('change', () => {
            if (autoShiftEnabled && autoShiftIntervalId === null) {
                autoShiftIntervalId = setInterval(autoShiftTick, AUTO_SHIFT_INTERVAL_MS);
            }
        });
    }

    // Couple checkbox
    document.getElementById('timerankCouple').addEventListener('change', (e) => {
        timerankCoupled = e.target.checked;
    });

    // Play checkbox for auto-advancing through TimeRank
    document.getElementById('timerankPlay').addEventListener('change', (e) => {
        if (e.target.checked) {
            startAutoShift();
        } else {
            stopAutoShift();
        }
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    for (const key of ['wt', 'ko']) {
        const vp = viewports[key];

        if (autoRotateEnabled && vp.controls && vp.controls.target) {
            const rotSpeed = 0.002;
            const offset = new THREE.Vector3().subVectors(vp.camera.position, vp.controls.target);
            const cos = Math.cos(rotSpeed);
            const sin = Math.sin(rotSpeed);
            const x = offset.x * cos - offset.z * sin;
            const z = offset.x * sin + offset.z * cos;
            offset.x = x;
            offset.z = z;
            vp.camera.position.copy(vp.controls.target).add(offset);
            vp.camera.lookAt(vp.controls.target);
        }

        vp.controls.update();
        vp.renderer.render(vp.scene, vp.camera);
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    setupThemeToggle();
    setupRightPanelResizer();
    initScene();
    loadData();
});
