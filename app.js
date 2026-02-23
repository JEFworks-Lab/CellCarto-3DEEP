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
    
// Column configuration - users can manually change these lists
const COORD_COLUMN_OPTIONS = ['x', 'y', 'z', 'transformedX', 'transformedY', 'transformedZ', 'Adj_transformedZ', 'X_shifted', 'TimeRank'];
let selectedCoordX = 'transformedX';
let selectedCoordY = 'transformedZ';
let selectedCoordZ = 'transformedY';
const column_names_categorical = ['Structure', 'HF', 'Sample', 'Group', 'CellType', 'Gene'];
const column_names_continuous = ['Time'];

// Global variables
let scene, camera, renderer, controls;
let points, pointCloud;
let allData = [];
let visibleIndices = null; // Will be Uint32Array
let colorMap = new Map();
let attributeValues = {}; // Will be dynamically populated
let continuousRanges = {}; // Will store min/max for each continuous variable
let cameraInitialized = false;
let activeFilters = []; // Array of filter objects: { attribute, type: 'categorical'|'time', values/range }
let initialCameraState = { position: null, target: null }; // Store initial camera state for reset
let renderedIndicesMap = null; // Map from instance index to data index for hover detection
let autoRotateEnabled = false; // Auto-rotation state
let highlightSphere = null; // Sphere to highlight hovered point
let tooltip = null; // Tooltip element
let isShiftPressed = false; // Track SHIFT key state
let raycaster = new THREE.Raycaster(); // For point picking
let mouse = new THREE.Vector2(); // Mouse position for raycasting
let eventListenersInitialized = false; // Track if event listeners have been set up

// Lazy loading state
let parquetBuffers = []; // Store downloaded buffers for lazy loading columns
let loadedColumns = new Set(); // Track which columns have been loaded
let isLoadingColumn = false; // Prevent concurrent column loads

// Progressive shard loading state
let loadedShardCount = 0; // How many shards are currently loaded
let isLoadingShards = false; // Prevent concurrent shard loads

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
    // Completely disable OrbitControls pan and rotate - we'll handle it custom
    controls.enableRotate = false;
    controls.enablePan = false; // We'll handle panning manually
    controls.enableZoom = true; // Enable zooming (mouse wheel)
    controls.zoomSpeed = 3.0; // Faster zoom (default is 1.0)
    controls.screenSpacePanning = false;
    controls.panSpeed = 1.0; // Pan speed reference
    
    // Disable OrbitControls mouse/touch handlers by overriding them
    controls.mouseButtons = {
        LEFT: null,  // Disable left mouse button
        MIDDLE: null, // Disable middle mouse button
        RIGHT: null  // Disable right mouse button
    };
    
    // Disable touch controls
    controls.touches = {
        ONE: null,
        TWO: null
    };
    
    // Track Control key state and mouse state
    let isControlPressed = false;
    let isDragging = false;
    let lastMousePosition = new THREE.Vector2();
    const panSpeed = 0.1; // Pan speed multiplier (increased for better responsiveness)
    const rotationSpeed = 0.01; // Rotation speed (increased for better responsiveness)
    
    // Listen for Control key
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Control' || event.ctrlKey) {
            if (!isControlPressed) {
                isControlPressed = true;
                console.log('[Camera Controls] Control key pressed - rotation mode enabled');
            }
        }
    });
    
    window.addEventListener('keyup', (event) => {
        if (event.key === 'Control' || !event.ctrlKey) {
            if (isControlPressed) {
                isControlPressed = false;
                console.log('[Camera Controls] Control key released - pan mode enabled');
            }
        }
    });
    
    // Custom mouse handling for panning and rotation
    const canvasElement = renderer.domElement;
    
    if (!canvasElement) {
        console.error('[Camera Controls] ERROR: Canvas element not found!');
        return;
    }
    
    console.log('[Camera Controls] Setting up event listeners on canvas:', {
        canvasElement: canvasElement,
        canvasId: canvasElement.id,
        canvasTag: canvasElement.tagName,
        canvasClasses: canvasElement.className,
        parentElement: canvasElement.parentElement?.id || 'none'
    });
    
    // Use pointer events (better for Mac trackpads) and mouse events as fallback
    const handlePointerDown = (event) => {
        // Only handle if clicking on canvas or its children
        const target = event.target;
        if (!canvasElement.contains(target) && target !== canvasElement) {
            return;
        }
        
        console.log('[Camera Controls] pointerdown/mousedown event:', {
            type: event.type,
            pointerId: event.pointerId,
            button: event.button,
            buttons: event.buttons,
            clientX: event.clientX,
            clientY: event.clientY,
            isControlPressed: isControlPressed,
            ctrlKey: event.ctrlKey,
            metaKey: event.metaKey,
            target: target?.tagName,
            targetId: target?.id
        });
        
        // Accept any pointer/mouse down on the canvas (Mac trackpads work with pointer events)
        if (event.button === 0 || event.button === 2 || event.buttons > 0 || event.pointerType === 'mouse' || event.pointerType === 'touch') {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            isDragging = true;
            lastMousePosition.set(event.clientX, event.clientY);
            console.log('[Camera Controls] Dragging started, mode:', isControlPressed ? 'ROTATE' : 'PAN');
        }
    };
    
    // Add both pointer and mouse events for maximum compatibility
    canvasElement.addEventListener('pointerdown', handlePointerDown, { capture: true, passive: false });
    canvasElement.addEventListener('mousedown', handlePointerDown, { capture: true, passive: false });
    
    // Also try on document as fallback
    document.addEventListener('pointerdown', (event) => {
        if (canvasElement.contains(event.target) || event.target === canvasElement) {
            handlePointerDown(event);
        }
    }, { capture: true, passive: false });
    
    document.addEventListener('mousedown', (event) => {
        if (canvasElement.contains(event.target) || event.target === canvasElement) {
            handlePointerDown(event);
        }
    }, { capture: true, passive: false });
    
    // Also handle touch events for Mac trackpads
    canvasElement.addEventListener('touchstart', (event) => {
        console.log('[Camera Controls] touchstart event:', {
            touches: event.touches.length,
            clientX: event.touches[0]?.clientX,
            clientY: event.touches[0]?.clientY,
            isControlPressed: isControlPressed
        });
        
        if (event.touches.length === 1) { // Single finger touch
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            isDragging = true;
            lastMousePosition.set(event.touches[0].clientX, event.touches[0].clientY);
            console.log('[Camera Controls] Touch dragging started, mode:', isControlPressed ? 'ROTATE' : 'PAN');
        }
    }, { capture: true, passive: false });
    
    const handlePointerMove = (event) => {
        if (isDragging) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const deltaX = event.clientX - lastMousePosition.x;
            const deltaY = event.clientY - lastMousePosition.y;
            
            console.log('[Camera Controls] mousemove while dragging:', {
                deltaX: deltaX.toFixed(2),
                deltaY: deltaY.toFixed(2),
                isControlPressed: isControlPressed,
                mode: isControlPressed ? 'ROTATE' : 'PAN'
            });
            
            if (isControlPressed) {
                // Control + drag: Rotate camera
                // Up/down: rotate around x-axis (pitch) to view z variation
                // Left/right: rotate around y-axis (yaw)
                
                console.log('[Camera Controls] ROTATING - Control held');
                
                // Get camera's current position relative to target
                const offset = new THREE.Vector3();
                offset.subVectors(camera.position, controls.target);
                
                const beforePos = camera.position.clone();
                
                // Apply rotations
                if (Math.abs(deltaY) > 0) {
                    // Rotate around world x-axis (pitch) - tilt to see z variation
                    const pitchAngle = deltaY * rotationSpeed;
                    console.log('[Camera Controls] Rotating around X-axis (pitch):', pitchAngle.toFixed(4));
                    const xAxis = new THREE.Vector3(1, 0, 0);
                    const rotationMatrixX = new THREE.Matrix4();
                    rotationMatrixX.makeRotationAxis(xAxis, pitchAngle);
                    offset.applyMatrix4(rotationMatrixX);
                }
                
                if (Math.abs(deltaX) > 0) {
                    // Rotate around world y-axis (yaw)
                    const yawAngle = deltaX * rotationSpeed;
                    console.log('[Camera Controls] Rotating around Y-axis (yaw):', yawAngle.toFixed(4));
                    const yAxis = new THREE.Vector3(0, 1, 0);
                    const rotationMatrixY = new THREE.Matrix4();
                    rotationMatrixY.makeRotationAxis(yAxis, yawAngle);
                    offset.applyMatrix4(rotationMatrixY);
                }
                
                // Update camera position
                camera.position.copy(controls.target).add(offset);
                
                // Update camera to look at target
                camera.lookAt(controls.target);
                
                console.log('[Camera Controls] Camera position updated:', {
                    before: `(${beforePos.x.toFixed(2)}, ${beforePos.y.toFixed(2)}, ${beforePos.z.toFixed(2)})`,
                    after: `(${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)})`
                });
            } else {
                // Default drag: Pan camera in x-y plane
                // Up/down: pan in y direction (world space)
                // Left/right: pan in x direction (world space)
                
                console.log('[Camera Controls] PANNING - No Control key');
                
                // Scale pan speed by camera distance for more natural feel
                const cameraDistance = camera.position.distanceTo(controls.target);
                const scaledPanSpeed = panSpeed * (cameraDistance * 0.01);
                
                const beforeTarget = controls.target.clone();
                const beforePos = camera.position.clone();
                
                if (Math.abs(deltaY) > 0) {
                    // Pan in y direction (world space)
                    // Positive for Google Maps-like "grab and drag" behavior (screen Y is inverted from world Y)
                    const yPanAmount = deltaY * scaledPanSpeed;
                    console.log('[Camera Controls] Panning in Y direction:', yPanAmount.toFixed(4));
                    const yPanVector = new THREE.Vector3(0, yPanAmount, 0);
                    controls.target.add(yPanVector);
                    camera.position.add(yPanVector);
                }
                
                if (Math.abs(deltaX) > 0) {
                    // Pan in x direction (world space)
                    // Negate for Google Maps-like "grab and drag" behavior
                    const xPanAmount = -deltaX * scaledPanSpeed;
                    console.log('[Camera Controls] Panning in X direction:', xPanAmount.toFixed(4));
                    const xPanVector = new THREE.Vector3(xPanAmount, 0, 0);
                    controls.target.add(xPanVector);
                    camera.position.add(xPanVector);
                }
                
                console.log('[Camera Controls] Target updated:', {
                    before: `(${beforeTarget.x.toFixed(2)}, ${beforeTarget.y.toFixed(2)}, ${beforeTarget.z.toFixed(2)})`,
                    after: `(${controls.target.x.toFixed(2)}, ${controls.target.y.toFixed(2)}, ${controls.target.z.toFixed(2)})`
                });
            }
            
            controls.update();
            camera.updateMatrixWorld();
            lastMousePosition.set(event.clientX, event.clientY);
        }
    };
    
    // Add both pointer and mouse events
    canvasElement.addEventListener('pointermove', handlePointerMove, { capture: true, passive: false });
    canvasElement.addEventListener('mousemove', handlePointerMove, { capture: true, passive: false });
    
    // Also on document as fallback
    document.addEventListener('pointermove', (event) => {
        if (isDragging) {
            handlePointerMove(event);
        }
    }, { capture: true, passive: false });
    
    document.addEventListener('mousemove', (event) => {
        if (isDragging) {
            handlePointerMove(event);
        }
    }, { capture: true, passive: false });
    
    // Handle touchmove for Mac trackpads
    canvasElement.addEventListener('touchmove', (event) => {
        if (isDragging && event.touches.length === 1) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const deltaX = event.touches[0].clientX - lastMousePosition.x;
            const deltaY = event.touches[0].clientY - lastMousePosition.y;
            
            console.log('[Camera Controls] touchmove while dragging:', {
                deltaX: deltaX.toFixed(2),
                deltaY: deltaY.toFixed(2),
                isControlPressed: isControlPressed,
                mode: isControlPressed ? 'ROTATE' : 'PAN'
            });
            
            // Use the same logic as mousemove
            if (isControlPressed) {
                // Rotate logic (same as mousemove)
                const offset = new THREE.Vector3();
                offset.subVectors(camera.position, controls.target);
                
                if (Math.abs(deltaY) > 0) {
                    const pitchAngle = deltaY * rotationSpeed;
                    const xAxis = new THREE.Vector3(1, 0, 0);
                    const rotationMatrixX = new THREE.Matrix4();
                    rotationMatrixX.makeRotationAxis(xAxis, pitchAngle);
                    offset.applyMatrix4(rotationMatrixX);
                }
                
                if (Math.abs(deltaX) > 0) {
                    const yawAngle = deltaX * rotationSpeed;
                    const yAxis = new THREE.Vector3(0, 1, 0);
                    const rotationMatrixY = new THREE.Matrix4();
                    rotationMatrixY.makeRotationAxis(yAxis, yawAngle);
                    offset.applyMatrix4(rotationMatrixY);
                }
                
                camera.position.copy(controls.target).add(offset);
                camera.lookAt(controls.target);
            } else {
                // Pan logic (same as mousemove)
                // Google Maps-like "grab and drag" behavior
                const cameraDistance = camera.position.distanceTo(controls.target);
                const scaledPanSpeed = panSpeed * (cameraDistance * 0.01);
                
                if (Math.abs(deltaY) > 0) {
                    const yPanAmount = deltaY * scaledPanSpeed;
                    const yPanVector = new THREE.Vector3(0, yPanAmount, 0);
                    controls.target.add(yPanVector);
                    camera.position.add(yPanVector);
                }
                
                if (Math.abs(deltaX) > 0) {
                    const xPanAmount = -deltaX * scaledPanSpeed;
                    const xPanVector = new THREE.Vector3(xPanAmount, 0, 0);
                    controls.target.add(xPanVector);
                    camera.position.add(xPanVector);
                }
            }
            
            controls.update();
            camera.updateMatrixWorld();
            lastMousePosition.set(event.touches[0].clientX, event.touches[0].clientY);
        }
    }, { capture: true, passive: false });
    
    const handlePointerUp = (event) => {
        console.log('[Camera Controls] pointerup/mouseup event:', {
            type: event.type,
            button: event.button,
            buttons: event.buttons,
            isDragging: isDragging,
            pointerId: event.pointerId
        });
        
        if (isDragging) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            isDragging = false;
            console.log('[Camera Controls] Dragging stopped');
        }
    };
    
    // Add both pointer and mouse events
    canvasElement.addEventListener('pointerup', handlePointerUp, { capture: true, passive: false });
    canvasElement.addEventListener('mouseup', handlePointerUp, { capture: true, passive: false });
    
    // Also on document as fallback
    document.addEventListener('pointerup', handlePointerUp, { capture: true, passive: false });
    document.addEventListener('mouseup', handlePointerUp, { capture: true, passive: false });
    
    // Handle touchend for Mac trackpads
    canvasElement.addEventListener('touchend', (event) => {
        console.log('[Camera Controls] touchend event');
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        isDragging = false;
        console.log('[Camera Controls] Touch dragging stopped');
    }, { capture: true, passive: false });
    
    // Handle pointer cancel (Mac trackpads sometimes send this)
    canvasElement.addEventListener('pointercancel', (event) => {
        console.log('[Camera Controls] pointercancel event');
        isDragging = false;
    }, { capture: true, passive: false });
    
    canvasElement.addEventListener('mouseleave', () => {
        if (isDragging) {
            console.log('[Camera Controls] Mouse left canvas while dragging - stopping drag');
            isDragging = false;
        }
    });
    
    canvasElement.addEventListener('pointerleave', () => {
        if (isDragging) {
            console.log('[Camera Controls] Pointer left canvas while dragging - stopping drag');
            isDragging = false;
        }
    });
    
    // Test listener to see if ANY events are being received on the canvas
    const testAllEvents = ['click', 'mousedown', 'pointerdown', 'touchstart', 'contextmenu'];
    testAllEvents.forEach(eventType => {
        canvasElement.addEventListener(eventType, (event) => {
            console.log(`[Camera Controls] TEST - ${eventType} event received on canvas:`, {
                type: event.type,
                target: event.target?.tagName,
                button: event.button,
                buttons: event.buttons
            });
        }, { capture: true });
    });
    
    // Also handle mouseup on window to catch cases where mouse is released outside canvas
    window.addEventListener('mouseup', (event) => {
        if (event.button === 0 || event.button === 2) {
            if (isDragging) {
                console.log('[Camera Controls] Mouse released outside canvas - stopping drag');
                isDragging = false;
            }
        }
    });
    
    // Add initial state logging
    console.log('[Camera Controls] Initialized:', {
        enableRotate: controls.enableRotate,
        enablePan: controls.enablePan,
        enableZoom: controls.enableZoom,
        canvasElement: canvasElement ? 'found' : 'NOT FOUND'
    });
    
    // Prevent context menu on right click when Control is held
    canvasElement.addEventListener('contextmenu', (event) => {
        if (isControlPressed) {
            event.preventDefault();
        }
    });
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
    
    // Track SHIFT key for hover highlighting
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Shift' || event.shiftKey) {
            if (!isShiftPressed) {
                isShiftPressed = true;
                console.log('[Hover] SHIFT pressed - hover mode enabled');
            }
        }
    });
    
    window.addEventListener('keyup', (event) => {
        if (event.key === 'Shift' || !event.shiftKey) {
            if (isShiftPressed) {
                isShiftPressed = false;
                console.log('[Hover] SHIFT released - hover mode disabled');
                // Hide highlight and tooltip when SHIFT is released
                hideHighlight();
            }
        }
    });
    
    // Create tooltip element
    tooltip = document.createElement('div');
    tooltip.id = 'point-tooltip';
    tooltip.style.cssText = `
        position: fixed;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        z-index: 10000;
        border: 1px solid rgba(255, 255, 255, 0.3);
        display: none;
        max-width: 300px;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
    `;
    document.body.appendChild(tooltip);
    console.log('[Hover] Tooltip element created');
    
    // Create highlight sphere (will be positioned dynamically)
    // Use a slightly larger sphere with wireframe for white border effect
    const highlightGeometry = new THREE.SphereGeometry(1, 16, 16);
    const highlightMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: false,
        opacity: 1.0,
        wireframe: true,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: false // Don't write to depth buffer
    });
    highlightSphere = new THREE.Mesh(highlightGeometry, highlightMaterial);
    highlightSphere.renderOrder = 1000; // Render on top
    highlightSphere.visible = false;
    scene.add(highlightSphere);
    
    console.log('[Hover] Highlight sphere created and added to scene');
    
    // Add mouse move handler for hover detection (reuse canvasElement from above)
    const hoverCanvasElement = renderer.domElement;
    hoverCanvasElement.addEventListener('mousemove', handleHover, { passive: true });
    hoverCanvasElement.addEventListener('pointermove', handleHover, { passive: true });
    
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

// Handle hover detection when SHIFT is held
function handleHover(event) {
    // Only handle hover if SHIFT is pressed and not dragging camera
    if (!isShiftPressed || !pointCloud || !renderedIndicesMap) {
        hideHighlight();
        return;
    }
    
    // Don't interfere with camera controls - check if mouse buttons are pressed
    if (event.buttons && event.buttons > 0) {
        // User is dragging, don't show hover
        hideHighlight();
        return;
    }
    
    const canvas = renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Update raycaster
    const pointSize = parseFloat(document.getElementById('pointSize')?.value || 1);
    raycaster.setFromCamera(mouse, camera);
    
    // Set raycaster params for Points (threshold is the picking radius)
    raycaster.params.Points.threshold = pointSize * 5;
    
    // Check intersection with Points object
    try {
        const intersects = raycaster.intersectObject(pointCloud, false);
        
        if (intersects.length > 0) {
            const intersection = intersects[0];
            const pointIndex = intersection.index; // For Points, use index instead of instanceId
            
            if (pointIndex !== undefined && pointIndex !== null && pointIndex < renderedIndicesMap.length) {
                const dataIdx = renderedIndicesMap[pointIndex];
                if (dataIdx !== undefined && dataIdx < allData.length) {
                    const point = allData[dataIdx];
                    
                    // Show highlight at the actual point position (not intersection point)
                    const pointPosition = new THREE.Vector3(point.x, point.y, point.z);
                    showHighlight(pointPosition, point, event.clientX, event.clientY);
                    
                    // Log for debugging (only occasionally to avoid spam)
                    if (Math.random() < 0.01) { // Log 1% of the time
                        console.log('[Hover] Point highlighted:', {
                            pointIndex,
                            dataIdx,
                            position: `(${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`
                        });
                    }
                } else {
                    hideHighlight();
                }
            } else {
                hideHighlight();
            }
        } else {
            hideHighlight();
        }
    } catch (error) {
        console.warn('[Hover] Error in raycasting:', error);
        hideHighlight();
    }
}

// Show highlight and tooltip for a point
function showHighlight(position, point, mouseX, mouseY) {
    if (!highlightSphere || !tooltip) {
        console.warn('[Hover] Highlight sphere or tooltip not available');
        return;
    }
    
    // Show and position highlight sphere
    const pointSize = parseFloat(document.getElementById('pointSize')?.value || 1);
    // Make highlight 1.5x the point size for visibility
    const highlightScale = pointSize * 1.5;
    highlightSphere.scale.set(highlightScale, highlightScale, highlightScale);
    highlightSphere.position.copy(position);
    highlightSphere.visible = true;
    
    // Create tooltip content
    const colorBy = document.getElementById('colorBy')?.value || 'gene';
    let tooltipContent = '<div style="font-weight: bold; margin-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 4px;">Point Information</div>';
    
    // Add coordinates with selected column names
    tooltipContent += `<div style="margin-bottom: 4px;"><strong>${selectedCoordX}:</strong> ${point.x.toFixed(2)}, <strong>${selectedCoordY}:</strong> ${point.y.toFixed(2)}, <strong>${selectedCoordZ}:</strong> ${point.z.toFixed(2)}</div>`;
    
    // Add all categorical attributes
    column_names_categorical.forEach(col => {
        if (point[col] !== undefined && point[col] !== '') {
            tooltipContent += `<div style="margin-bottom: 2px;"><strong>${col}:</strong> ${point[col]}</div>`;
        }
    });
    
    // Add all continuous attributes
    column_names_continuous.forEach(col => {
        if (point[col] !== null && point[col] !== undefined) {
            tooltipContent += `<div style="margin-bottom: 2px;"><strong>${col}:</strong> ${point[col].toFixed(4)}</div>`;
        }
    });
    
    tooltip.innerHTML = tooltipContent;
    tooltip.style.display = 'block';
    
    // Position tooltip near mouse cursor
    const offset = 15;
    tooltip.style.left = (mouseX + offset) + 'px';
    tooltip.style.top = (mouseY + offset) + 'px';
    
    // Adjust if tooltip goes off screen
    requestAnimationFrame(() => {
        const tooltipRect = tooltip.getBoundingClientRect();
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        
        if (tooltipRect.right > windowWidth) {
            tooltip.style.left = (mouseX - tooltipRect.width - offset) + 'px';
        }
        if (tooltipRect.bottom > windowHeight) {
            tooltip.style.top = (mouseY - tooltipRect.height - offset) + 'px';
        }
        if (tooltipRect.left < 0) {
            tooltip.style.left = offset + 'px';
        }
        if (tooltipRect.top < 0) {
            tooltip.style.top = offset + 'px';
        }
    });
}

// Hide highlight and tooltip
function hideHighlight() {
    if (highlightSphere) {
        highlightSphere.visible = false;
    }
    if (tooltip) {
        tooltip.style.display = 'none';
    }
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
            
            // Set display coordinates from selected columns
            point.x = point['_raw_' + selectedCoordX] ?? 0;
            point.y = point['_raw_' + selectedCoordY] ?? 0;
            point.z = point['_raw_' + selectedCoordZ] ?? 0;
            
            // Add loaded columns by index
            for (const col of loadedColumns) {
                const idx = colIdx[col];
                if (idx === undefined) continue;
                
                if (col === 'Time') {
                    point[col] = row[idx];
                } else if (column_names_categorical.includes(col)) {
                    const value = row[idx];
                    point[col] = value != null ? String(value) : '';
                    if (value && attributeValues[col]) {
                        attributeValues[col].add(point[col]);
                    }
                }
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
        
        // Recreate point cloud with new data
        createPointCloud();
        
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
        // - Time (continuous attribute)
        const defaultColorBy = column_names_categorical[0]; // 'Structure'
        const essentialColumns = [
            ...COORD_COLUMN_OPTIONS,
            'Time',
            defaultColorBy
        ];
        
        loadingText.textContent = 'Parsing essential columns...';
        await new Promise(resolve => setTimeout(resolve, 0));
        
        // Parse only essential columns
        const rows = await parseParquetColumns(parquetBuffers, essentialColumns, (current, total) => {
            loadingText.textContent = `Parsing file ${current}/${total}...`;
        });
        
        // Mark these columns as loaded
        essentialColumns.forEach(col => loadedColumns.add(col));
        
        // Create column index map (hyparquet returns arrays, not objects)
        // The array indices correspond to the order of columns we requested
        const colIdx = {};
        essentialColumns.forEach((col, idx) => {
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
                
                // Set display coordinates from selected columns
                point.x = point['_raw_' + selectedCoordX] ?? 0;
                point.y = point['_raw_' + selectedCoordY] ?? 0;
                point.z = point['_raw_' + selectedCoordZ] ?? 0;
                
                // Add the default colorBy attribute (already loaded)
                const rowDefaultColorBy = row[colIdx[defaultColorBy]];
                point[defaultColorBy] = rowDefaultColorBy != null ? String(rowDefaultColorBy) : '';
                if (rowDefaultColorBy) {
                    if (!attributeValues[defaultColorBy]) attributeValues[defaultColorBy] = new Set();
                    attributeValues[defaultColorBy].add(point[defaultColorBy]);
                }
                
                // Add Time (continuous attribute)
                const rowTime = row[colIdx.Time];
                point.Time = rowTime;
                if (rowTime != null) {
                    if (!continuousRanges.Time) continuousRanges.Time = { min: Infinity, max: -Infinity };
                    continuousRanges.Time.min = Math.min(continuousRanges.Time.min, rowTime);
                    continuousRanges.Time.max = Math.max(continuousRanges.Time.max, rowTime);
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
        const allAttributes = [...column_names_categorical, ...column_names_continuous];
        allAttributes.forEach(attr => {
            const option = document.createElement('option');
            option.value = attr;
            option.textContent = attr;
            colorBySelect.appendChild(option);
        });
        if (allAttributes.length > 0) {
            colorBySelect.value = allAttributes[0];
        }
        
        // Build filter UI
        renderFilters();
        
        // Create initial visualization
        loadingText.textContent = 'Rendering visualization...';
        createPointCloud();
        
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
        
        // Re-render the point cloud with new colors
        if (pointCloud) {
            createPointCloud();
        }
        
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
    
    // Re-render the visualization
    if (pointCloud) {
        createPointCloud();
    }
    
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
                legendDiv.innerHTML = `<div class="legend-label">No ${colorBy} data</div>`;
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
function setCameraToXYPlaneView(geometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = box.getSize(new THREE.Vector3());
    
    // Calculate the extent in x-y plane
    const xyExtent = Math.max(size.x, size.y);
    
    // Position camera above the center (along z-axis), looking down at x-y plane
    // Calculate distance needed to see the full x-y extent
    const fovRad = camera.fov * (Math.PI / 180);
    
    // Calculate distance needed to fit the larger of x or y extent
    // For perspective camera: height_visible = 2 * distance * tan(fov/2)
    const halfHeight = xyExtent / 2;
    const distance = halfHeight / Math.tan(fovRad / 2);
    
    // Add some padding (10% margin)
    const cameraHeight = Math.max(distance * 1.1, size.z * 1.5, 100);
    
    const cameraPosition = new THREE.Vector3(
        center.x,
        center.y,
        center.z + cameraHeight
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
    // Remove existing point cloud
    if (pointCloud) {
        scene.remove(pointCloud);
        pointCloud.geometry.dispose();
        pointCloud.material.dispose();
    }
    
    // Hide highlight when recreating point cloud
    if (highlightSphere) {
        highlightSphere.visible = false;
    }
    
    if (!visibleIndices || visibleIndices.length === 0) {
        document.getElementById('visibleCount').textContent = 'Visible points: 0';
        return;
    }
    
    const colorBy = document.getElementById('colorBy').value;
    const pointSize = parseFloat(document.getElementById('pointSize').value);
    
    // Since shards are pre-shuffled random samples, just render up to MAX_POINTS
    // No additional sampling needed - the data itself is already a random sample
    const visibleCount = visibleIndices.length;
    const renderCount = Math.min(visibleCount, MAX_POINTS);
    
    // Take the first renderCount points (they're already randomly ordered from shuffle)
    const indicesToRender = visibleIndices.subarray(0, renderCount);
    
    const count = indicesToRender.length;
    
    // Create BufferGeometry for THREE.Points (much more efficient than InstancedMesh with spheres)
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
        const dataIdx = indicesToRender[i];
        const point = allData[dataIdx];
        
        // Set position
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
        
        // Set color
        const valueForColor = point[colorBy];
        const pointColor = getColorForValue(valueForColor, colorBy);
        colors[i * 3] = pointColor.r;
        colors[i * 3 + 1] = pointColor.g;
        colors[i * 3 + 2] = pointColor.b;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Create ShaderMaterial for circular points with smooth edges
    const material = new THREE.ShaderMaterial({
        uniforms: {
            pointSize: { value: pointSize * 2 },
            opacity: { value: 0.8 }
        },
        vertexShader: `
            attribute vec3 color;
            varying vec3 vColor;
            uniform float pointSize;
            
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = pointSize * (300.0 / -mvPosition.z); // Size attenuation
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            uniform float opacity;
            
            void main() {
                // Calculate distance from center of point (gl_PointCoord is 0-1)
                vec2 center = gl_PointCoord - vec2(0.5);
                float dist = length(center);
                
                // Discard pixels outside the circle
                if (dist > 0.5) discard;
                
                // Smooth edge (anti-aliasing)
                float alpha = opacity * (1.0 - smoothstep(0.45, 0.5, dist));
                
                gl_FragColor = vec4(vColor, alpha);
            }
        `,
        transparent: true,
        depthWrite: false, // Better blending for transparent points
    });
    
    // Create the Points object
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
    
    // Store mapping from instance index to data index for hover detection
    renderedIndicesMap = indicesToRender;
    
    // Update legend
    updateLegend();
    
    // Update info
    const isCapped = indicesToRender.length < visibleIndices.length;
    document.getElementById('visibleCount').textContent = 
        `Rendering: ${indicesToRender.length.toLocaleString()} points${isCapped ? ` (capped at ${MAX_POINTS.toLocaleString()})` : ''}`;
    
    // Auto-adjust camera for x-y plane view only on first render
    if (!cameraInitialized) {
        const tempGeometry = new THREE.BufferGeometry();
        tempGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        tempGeometry.computeBoundingBox();
        setCameraToXYPlaneView(tempGeometry);
        tempGeometry.dispose();
        cameraInitialized = true;
    }
}

// Create a filter UI element
function createFilterElement(filterId, attribute) {
    const filterDiv = document.createElement('div');
    filterDiv.className = 'filter-block';
    filterDiv.dataset.filterId = filterId;
    
    const availableAttributes = [...column_names_categorical, ...column_names_continuous];
    const availableOptions = availableAttributes.map(attr => 
        `<option value="${attr}" ${attr === attribute ? 'selected' : ''}>${attr}</option>`
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
    
    // If no active filters, show all points
    if (activeFilters.length === 0) {
        visibleIndices = new Uint32Array(allData.length);
        for (let i = 0; i < allData.length; i++) {
            visibleIndices[i] = i;
        }
        console.log(`[Filter] No filters active - showing all ${visibleIndices.length} points`);
        createPointCloud();
        return;
    }
    
    // Start with all indices
    let candidateIndices = [];
    for (let i = 0; i < allData.length; i++) {
        candidateIndices.push(i);
    }
    
    // Apply each filter sequentially (AND logic)
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
    
    createPointCloud();
    // Legend updates automatically when point cloud is recreated
}

// Populate the coordinate axis dropdowns
function populateCoordDropdowns() {
    const xSelect = document.getElementById('coordX');
    const ySelect = document.getElementById('coordY');
    const zSelect = document.getElementById('coordZ');
    if (!xSelect || !ySelect || !zSelect) return;

    [xSelect, ySelect, zSelect].forEach((sel, axisIdx) => {
        sel.innerHTML = '';
        const defaultVal = [selectedCoordX, selectedCoordY, selectedCoordZ][axisIdx];
        COORD_COLUMN_OPTIONS.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (col === defaultVal) opt.selected = true;
            sel.appendChild(opt);
        });
    });
}

// Remap coordinates from stored raw column values
function remapCoordinates() {
    console.log(`[Coordinates] Remapping to X=${selectedCoordX}, Y=${selectedCoordY}, Z=${selectedCoordZ}`);

    for (let i = 0; i < allData.length; i++) {
        const point = allData[i];
        point.x = point['_raw_' + selectedCoordX] ?? 0;
        point.y = point['_raw_' + selectedCoordY] ?? 0;
        point.z = point['_raw_' + selectedCoordZ] ?? 0;
    }

    cameraInitialized = false;

    activeFilters = [];
    renderFilters();

    colorMap.clear();

    if (visibleIndices && visibleIndices.length > 0) {
        visibleIndices = new Uint32Array(allData.length);
        for (let i = 0; i < allData.length; i++) {
            visibleIndices[i] = i;
        }
        createPointCloud();
    }

    updateLegend();
}

// Setup event listeners
function setupEventListeners() {
    // Coordinate axis change handlers
    function handleCoordChange() {
        selectedCoordX = document.getElementById('coordX').value;
        selectedCoordY = document.getElementById('coordY').value;
        selectedCoordZ = document.getElementById('coordZ').value;
        remapCoordinates();
    }
    document.getElementById('coordX').addEventListener('change', handleCoordChange);
    document.getElementById('coordY').addEventListener('change', handleCoordChange);
    document.getElementById('coordZ').addEventListener('change', handleCoordChange);
    
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
        
        // Update colors
        if (pointCloud) {
            createPointCloud();
        } else {
            updateLegend();
        }
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
        // Update shader uniform directly for instant response (no need to recreate point cloud)
        if (pointCloud && pointCloud.material && pointCloud.material.uniforms) {
            pointCloud.material.uniforms.pointSize.value = newSize * 2;
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
                setCameraToXYPlaneView(tempGeometry);
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
    
    // Auto-rotate checkbox
    document.getElementById('autoRotate').addEventListener('change', (e) => {
        autoRotateEnabled = e.target.checked;
        console.log('[Camera] Auto-rotate:', autoRotateEnabled ? 'enabled' : 'disabled');
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Auto-rotation: rotate camera around the target
    if (autoRotateEnabled && controls && controls.target) {
        const rotationSpeed = 0.002; // Radians per frame
        
        // Get camera's offset from target
        const offset = new THREE.Vector3();
        offset.subVectors(camera.position, controls.target);
        
        // Rotate around Y axis
        const angle = rotationSpeed;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const x = offset.x * cos - offset.z * sin;
        const z = offset.x * sin + offset.z * cos;
        offset.x = x;
        offset.z = z;
        
        // Update camera position
        camera.position.copy(controls.target).add(offset);
        camera.lookAt(controls.target);
    }
    
    controls.update();
    renderer.render(scene, camera);
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    initScene();
    loadData();
});
