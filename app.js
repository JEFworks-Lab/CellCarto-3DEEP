import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global variables
let scene, camera, renderer, controls;
let points, pointCloud, sphereMesh;
let allData = [];
let visibleIndices = null; // Will be Uint32Array
let colorMap = new Map();
let attributeValues = {
    Gene: new Set(),
    CellType: new Set(),
    Structure: new Set(),
    cell: new Set(),
    Time: new Set(),
    HF: new Set()
};
let timeRange = { min: Infinity, max: -Infinity };
let cameraInitialized = false;
let activeFilters = []; // Array of filter objects: { attribute, type: 'categorical'|'time', values/range }
let initialCameraState = { position: null, target: null }; // Store initial camera state for reset

// Initialize Three.js scene
function initScene() {
    const container = document.getElementById('center-panel');
    const canvas = document.getElementById('scene');
    
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    
    // Camera
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 10000);
    camera.position.set(0, 0, 100);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 10000;
    controls.enableRotate = true;
    controls.enablePan = true; // Enable panning to allow moving the focus point
    controls.enableZoom = true; // Enable zooming
    controls.screenSpacePanning = false; // Pan perpendicular to camera view
    controls.panSpeed = 1.0; // Pan speed
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
    
    // Keep loading message visible initially (will be hidden after data loads)
}

function onWindowResize() {
    const container = document.getElementById('center-panel');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Helper function to load and decompress a gzipped file
async function loadGzippedFile(url) {
    const response = await fetch(url);
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} for ${url}`);
    }
    
    if (!response.body) {
        throw new Error('Response body is null');
    }
    
    const decompressionStream = new DecompressionStream('gzip');
    const decompressedStream = response.body.pipeThrough(decompressionStream);
    const decompressedResponse = new Response(decompressedStream);
    const text = await decompressedResponse.text();
    
    return text;
}

// Load and parse data with chunked processing
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
    loadingText.textContent = 'Loading data... This may take a moment.';
    
    try {
        // Load both parts in parallel
        loadingText.textContent = 'Loading data files (part 1/2)...';
        const loadStartTime = Date.now();
        
        const [text1, text2] = await Promise.all([
            loadGzippedFile('data/20241209.HF.Coordinate.Celltype.Structure.transformed.part01.txt.gz'),
            loadGzippedFile('data/20241209.HF.Coordinate.Celltype.Structure.transformed.part02.txt.gz')
        ]);
        
        const loadTime = ((Date.now() - loadStartTime) / 1000).toFixed(1);
        loadingText.textContent = `Files loaded (${loadTime}s). Processing data...`;
        await new Promise(resolve => setTimeout(resolve, 100)); // Brief pause to show message
        
        // Split into lines
        loadingText.textContent = 'Parsing data files...';
        const lines1 = text1.split('\n');
        const lines2 = text2.split('\n');
        
        // Get headers from first file (both should have same headers)
        const headers = lines1[0].split('\t');
        
        // Merge lines (skip header from second file, keep all from first)
        const allLines = [...lines1, ...lines2.slice(1)];
        const totalLines = allLines.length;
        
        // Find column indices
        const xIdx = headers.indexOf('transformedX');
        const yIdx = headers.indexOf('transformedY');
        const zIdx = headers.indexOf('transformedZ');
        const geneIdx = headers.indexOf('Gene');
        const cellTypeIdx = headers.indexOf('CellType');
        const structureIdx = headers.indexOf('Structure');
        const cellIdx = headers.indexOf('cell');
        const timeIdx = headers.indexOf('Time');
        const hfIdx = headers.indexOf('HF');
        
        // Pre-allocate arrays for better performance
        allData = [];
        allData.length = 0; // Clear but keep reference
        
        const CHUNK_SIZE = 50000; // Process in chunks to avoid blocking
        
        // Parse data in chunks
        for (let startIdx = 1; startIdx < totalLines; startIdx += CHUNK_SIZE) {
            const endIdx = Math.min(startIdx + CHUNK_SIZE, totalLines);
            
            for (let i = startIdx; i < endIdx; i++) {
                const line = allLines[i].trim();
                if (!line) continue;
                
                const cols = line.split('\t');
                if (cols.length < headers.length) continue;
                
                const x = parseFloat(cols[xIdx]);
                const y = parseFloat(cols[yIdx]);
                const z = parseFloat(cols[zIdx]);
                
                if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
                
                const timeValue = parseFloat(cols[timeIdx]);
                const timeNum = isNaN(timeValue) ? null : timeValue;
                
                const point = {
                    x: x,
                    y: y,
                    z: z,
                    Gene: cols[geneIdx] || '',
                    CellType: cols[cellTypeIdx] || '',
                    Structure: cols[structureIdx] || '',
                    cell: cols[cellIdx] || '',
                    Time: timeNum,
                    TimeStr: cols[timeIdx] || '',
                    HF: cols[hfIdx] || ''
                };
                
                allData.push(point);
                
                // Collect unique values for each attribute (categorical)
                if (point.Gene) attributeValues.Gene.add(point.Gene);
                if (point.CellType) attributeValues.CellType.add(point.CellType);
                if (point.Structure) attributeValues.Structure.add(point.Structure);
                if (point.cell) attributeValues.cell.add(point.cell);
                if (point.TimeStr) attributeValues.Time.add(point.TimeStr);
                if (point.HF) attributeValues.HF.add(point.HF);
                
                // Track Time range for continuous filtering
                if (timeNum !== null) {
                    timeRange.min = Math.min(timeRange.min, timeNum);
                    timeRange.max = Math.max(timeRange.max, timeNum);
                }
            }
            
            // Update progress and yield to browser
            if (endIdx % (CHUNK_SIZE * 5) === 0 || endIdx === totalLines) {
                const progress = ((endIdx / totalLines) * 100).toFixed(1);
                loadingText.textContent = `Parsing data... ${progress}%`;
                await new Promise(resolve => setTimeout(resolve, 0)); // Yield to browser
            }
        }
        
        loadingText.textContent = `Loaded ${allData.length.toLocaleString()} points. Initializing visualization...`;
        document.getElementById('pointCount').textContent = `Total points: ${allData.length.toLocaleString()}`;
        await new Promise(resolve => setTimeout(resolve, 100)); // Brief pause to show final message
        
        // Initialize visible indices more efficiently
        visibleIndices = new Uint32Array(allData.length);
        for (let i = 0; i < allData.length; i++) {
            visibleIndices[i] = i;
        }
        
        // Build filter UI
        renderFilters();
        
        // Create initial visualization
        loadingText.textContent = 'Rendering visualization...';
        createPointCloud();
        
        // Initialize legend
        updateLegend();
        
        // Set up event listeners
        setupEventListeners();
        
        // Start animation loop
        animate();
        
        // Hide loading message after a brief delay to show final render
        setTimeout(() => {
            loadingEl.style.display = 'none';
        }, 500);
        
    } catch (error) {
        console.error('Error loading data:', error);
        loadingText.textContent = 'Error loading data. Please check the console.';
        loadingEl.style.background = 'rgba(231, 76, 60, 0.9)';
        loadingEl.style.borderColor = 'rgba(192, 57, 43, 0.5)';
    }
}

// Generate color for a value
function getColorForValue(value, attribute) {
    if (value === null || value === undefined || value === '') return new THREE.Color(0x888888);
    
    // For Time (continuous), use a gradient based on the value
    if (attribute === 'Time' && typeof value === 'number') {
        const normalized = (value - timeRange.min) / (timeRange.max - timeRange.min);
        const color = new THREE.Color();
        // Use a color gradient from blue to red
        color.setHSL((1 - normalized) * 0.7, 0.8, 0.5);
        return color;
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

// Update the color legend
function updateLegend() {
    const legendDiv = document.getElementById('legend');
    if (!legendDiv) return;
    
    const colorBy = document.getElementById('colorBy').value;
    
    // Get unique values from visible data
    const visibleValues = new Set();
    if (visibleIndices && visibleIndices.length > 0) {
        for (let i = 0; i < visibleIndices.length; i++) {
            const point = allData[visibleIndices[i]];
            if (colorBy === 'Time') {
                if (point.Time !== null) {
                    visibleValues.add(point.Time);
                }
            } else {
                const value = point[colorBy] || '';
                if (value) {
                    visibleValues.add(value);
                }
            }
        }
    }
    
    legendDiv.innerHTML = '';
    
    if (colorBy === 'Time') {
        // Show gradient for continuous Time values
        if (visibleValues.size > 0) {
            const timeValues = Array.from(visibleValues).map(v => parseFloat(v)).filter(v => !isNaN(v));
            if (timeValues.length > 0) {
                const minTime = Math.min(...timeValues);
                const maxTime = Math.max(...timeValues);
                
                // Create gradient canvas
                const canvas = document.createElement('canvas');
                canvas.width = 200;
                canvas.height = 30;
                canvas.className = 'legend-gradient';
                const ctx = canvas.getContext('2d');
                
                const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
                for (let i = 0; i <= 100; i++) {
                    const normalized = i / 100;
                    const timeValue = minTime + (maxTime - minTime) * normalized;
                    const color = getColorForValue(timeValue, 'Time');
                    const stop = i / 100;
                    gradient.addColorStop(stop, `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`);
                }
                
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                legendDiv.appendChild(canvas);
                
                const labelsDiv = document.createElement('div');
                labelsDiv.className = 'legend-gradient-labels';
                labelsDiv.innerHTML = `<span>${minTime.toFixed(4)}</span><span>${maxTime.toFixed(4)}</span>`;
                legendDiv.appendChild(labelsDiv);
            } else {
                legendDiv.innerHTML = '<div class="legend-label">No Time data</div>';
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
            
            const colorDiv = document.createElement('div');
            colorDiv.className = 'legend-color';
            const color = getColorForValue(value, colorBy);
            colorDiv.style.backgroundColor = `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'legend-label';
            labelDiv.textContent = value || '(empty)';
            
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

// Set camera to x-z plane view
function setCameraToXZPlaneView(geometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = box.getSize(new THREE.Vector3());
    
    // Calculate the extent in x-z plane
    const xzExtent = Math.max(size.x, size.z);
    
    // Position camera above the center (along y-axis), looking at x-z plane
    // Calculate distance needed to see the full x-z extent
    const fovRad = camera.fov * (Math.PI / 180);
    
    // Calculate distance needed to fit the larger of x or z extent
    // For perspective camera: height_visible = 2 * distance * tan(fov/2)
    const halfHeight = xzExtent / 2;
    const distance = halfHeight / Math.tan(fovRad / 2);
    
    // Add some padding (10% margin)
    const cameraHeight = Math.max(distance * 1.1, size.y * 1.5, 100);
    
    const cameraPosition = new THREE.Vector3(
        center.x,
        center.y + cameraHeight,
        center.z
    );
    
    camera.position.copy(cameraPosition);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
    
    // Store initial state for reset
    initialCameraState.position = cameraPosition.clone();
    initialCameraState.target = center.clone();
    
    return { position: cameraPosition.clone(), target: center.clone() };
}

// Create point cloud with spheres
function createPointCloud() {
    // Remove existing point cloud/spheres
    if (pointCloud) {
        scene.remove(pointCloud);
        pointCloud.geometry.dispose();
        pointCloud.material.dispose();
    }
    if (sphereMesh) {
        scene.remove(sphereMesh);
        sphereMesh.geometry.dispose();
        sphereMesh.material.dispose();
    }
    
    if (!visibleIndices || visibleIndices.length === 0) {
        document.getElementById('visibleCount').textContent = 'Visible points: 0';
        return;
    }
    
    const colorBy = document.getElementById('colorBy').value;
    const pointSize = parseFloat(document.getElementById('pointSize').value);
    const showAll = document.getElementById('showAllPoints').checked;
    const sampleRate = Math.max(0.01, Math.min(1, parseInt(document.getElementById('sampleRate').value) / 100));
    
    // Sample data if needed - limit to reasonable number of points
    // For spheres, reduce max points slightly for performance
    const MAX_POINTS = 50000; // Reduced for sphere rendering performance
    let indicesToRender = [];
    const visibleCount = visibleIndices.length;
    
    if (showAll && visibleCount <= MAX_POINTS) {
        // Use all visible indices if within limit
        indicesToRender = Array.from(visibleIndices);
    } else {
        // Sample based on rate, but cap at MAX_POINTS
        const targetCount = Math.min(visibleCount * sampleRate, MAX_POINTS);
        const step = Math.max(1, Math.floor(visibleCount / targetCount));
        
        for (let i = 0; i < visibleCount && indicesToRender.length < MAX_POINTS; i += step) {
            indicesToRender.push(visibleIndices[i]);
        }
    }
    
    const count = indicesToRender.length;
    
    // Create sphere geometry for instancing
    const sphereGeometry = new THREE.SphereGeometry(pointSize, 8, 6); // radius, widthSegments, heightSegments
    
    // Create instanced mesh
    sphereMesh = new THREE.InstancedMesh(sphereGeometry, null, count);
    
    // Create material with instanced colors support
    const material = new THREE.MeshPhongMaterial({
        color: 0xffffff, // Base color (will be overridden by instance colors)
        transparent: true,
        opacity: 0.8,
        flatShading: true // Use flat shading for better performance
    });
    sphereMesh.material = material;
    
    // Enable instanced colors
    const colors = new Float32Array(count * 3);
    
    // Set up instance matrices and colors
    const matrix = new THREE.Matrix4();
    
    // Create geometry for bounding box calculation
    const positions = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
        const dataIdx = indicesToRender[i];
        const point = allData[dataIdx];
        
        // Set position
        matrix.makeTranslation(point.x, point.y, point.z);
        sphereMesh.setMatrixAt(i, matrix);
        
        // Set color
        const valueForColor = colorBy === 'Time' ? point.Time : point[colorBy];
        const pointColor = getColorForValue(valueForColor, colorBy);
        sphereMesh.setColorAt(i, pointColor);
        
        // Store position for bounding box
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
    }
    
    // Update instance matrices and colors
    sphereMesh.instanceMatrix.needsUpdate = true;
    if (sphereMesh.instanceColor) {
        sphereMesh.instanceColor.needsUpdate = true;
    }
    
    scene.add(sphereMesh);
    pointCloud = sphereMesh; // Keep reference for cleanup
    
    // Update legend
    updateLegend();
    
    // Update info
    document.getElementById('visibleCount').textContent = 
        `Visible points: ${indicesToRender.length.toLocaleString()}${!showAll && visibleIndices.length > MAX_POINTS ? ` (sampled from ${visibleIndices.length.toLocaleString()})` : ''}`;
    
    // Auto-adjust camera for x-z plane view only on first render
    if (!cameraInitialized) {
        const tempGeometry = new THREE.BufferGeometry();
        tempGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        tempGeometry.computeBoundingBox();
        setCameraToXZPlaneView(tempGeometry);
        tempGeometry.dispose();
        cameraInitialized = true;
    }
}

// Create a filter UI element
function createFilterElement(filterId, attribute) {
    const filterDiv = document.createElement('div');
    filterDiv.className = 'filter-block';
    filterDiv.dataset.filterId = filterId;
    
    const availableAttributes = ['Gene', 'CellType', 'Structure', 'cell', 'Time', 'HF'];
    const availableOptions = availableAttributes.map(attr => 
        `<option value="${attr}" ${attr === attribute ? 'selected' : ''}>${attr}</option>`
    ).join('');
    
    if (attribute === 'Time') {
        // Time filter with sliders
        const timeRangeSize = timeRange.max - timeRange.min;
        const stepSize = timeRangeSize > 0 ? Math.max(timeRangeSize / 1000, 0.0001) : 0.0001;
        
        filterDiv.innerHTML = `
            <div class="filter-header">
                <select class="filter-attribute" data-filter-id="${filterId}">
                    <option value="">Select attribute...</option>
                    ${availableOptions}
                </select>
                <button class="remove-filter" data-filter-id="${filterId}">×</button>
            </div>
            <div class="filter-content" data-filter-id="${filterId}">
                <label>Time Range:</label>
                <div class="time-slider-wrapper">
                    <input type="range" class="time-min" data-filter-id="${filterId}" 
                           min="${timeRange.min}" max="${timeRange.max}" 
                           step="${stepSize}" value="${timeRange.min}">
                    <input type="range" class="time-max" data-filter-id="${filterId}" 
                           min="${timeRange.min}" max="${timeRange.max}" 
                           step="${stepSize}" value="${timeRange.max}">
                </div>
                <div class="time-values">
                    <span>Min: <span class="time-min-value">${timeRange.min.toFixed(4)}</span></span>
                    <span>Max: <span class="time-max-value">${timeRange.max.toFixed(4)}</span></span>
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
        
        newSelect.addEventListener('change', (e) => {
            const filterId = e.target.dataset.filterId;
            const attribute = e.target.value;
            
            const filter = activeFilters.find(f => f.id === filterId);
            if (filter) {
                filter.attribute = attribute;
                if (attribute === 'Time') {
                    filter.type = 'time';
                    filter.range = { min: timeRange.min, max: timeRange.max };
                    filter.values = null;
                } else if (attribute) {
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
            activeFilters = activeFilters.filter(f => f.id !== filterId);
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
    // Start with all indices
    let candidateIndices = [];
    for (let i = 0; i < allData.length; i++) {
        candidateIndices.push(i);
    }
    
    // Apply each filter sequentially (AND logic)
    activeFilters.forEach(filter => {
        if (!filter.attribute || !filter.type) return;
        
        const filteredIndices = [];
        
        if (filter.type === 'time' && filter.range) {
            // Time range filter
            const minTime = filter.range.min;
            const maxTime = filter.range.max;
            
            for (let idx of candidateIndices) {
                const point = allData[idx];
                if (point.Time !== null && point.Time >= minTime && point.Time <= maxTime) {
                    filteredIndices.push(idx);
                }
            }
        } else if (filter.type === 'categorical' && filter.values && filter.values.size > 0) {
            // Categorical filter
            for (let idx of candidateIndices) {
                const point = allData[idx];
                if (filter.values.has(point[filter.attribute] || '')) {
                    filteredIndices.push(idx);
                }
            }
        } else {
            // Invalid filter, skip it (don't filter anything)
            return;
        }
        
        candidateIndices = filteredIndices;
    });
    
    // Convert to Uint32Array
    visibleIndices = new Uint32Array(candidateIndices);
    
    createPointCloud();
    // Legend updates automatically when point cloud is recreated
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('colorBy').addEventListener('change', () => {
        // Update colors immediately for color changes (no throttle needed, just recolor)
        if (pointCloud) {
            createPointCloud();
        } else {
            // If no point cloud yet, just update legend
            updateLegend();
        }
    });
    
    // Add filter button
    document.getElementById('addFilter').addEventListener('click', () => {
        const filterId = 'filter_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        activeFilters.push({
            id: filterId,
            attribute: '',
            type: null,
            values: null,
            range: null
        });
        renderFilters();
    });
    
    document.getElementById('pointSize').addEventListener('input', (e) => {
        document.getElementById('pointSizeValue').textContent = parseFloat(e.target.value).toFixed(1);
        // Recreate point cloud with new size
        if (visibleIndices && visibleIndices.length > 0) {
            throttleFilterUpdate();
        }
    });
    
    document.getElementById('sampleRate').addEventListener('input', (e) => {
        document.getElementById('sampleRateValue').textContent = e.target.value + '%';
        throttleFilterUpdate();
    });
    
    document.getElementById('showAllPoints').addEventListener('change', () => {
        throttleFilterUpdate();
    });
    
    
    document.getElementById('resetCamera').addEventListener('click', () => {
        if (pointCloud) {
            // Recalculate bounding box from current visible data
            const positions = [];
            if (visibleIndices && visibleIndices.length > 0) {
                const MAX_SAMPLE = 10000; // Sample points for bounding box calculation
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
                setCameraToXZPlaneView(tempGeometry);
                tempGeometry.dispose();
            } else if (initialCameraState.position && initialCameraState.target) {
                // Fall back to stored initial state
                camera.position.copy(initialCameraState.position);
                controls.target.copy(initialCameraState.target);
                camera.lookAt(initialCameraState.target);
                controls.update();
            }
        } else if (initialCameraState.position && initialCameraState.target) {
            // Use stored initial state if no point cloud yet
            camera.position.copy(initialCameraState.position);
            controls.target.copy(initialCameraState.target);
            camera.lookAt(initialCameraState.target);
            controls.update();
        }
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    initScene();
    loadData();
});