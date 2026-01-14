import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// File paths - users can manually change this list
const files = [
    'data/20260113.hairfollicle.part01.tsv.gz',
    'data/20260113.hairfollicle.part02.tsv.gz',
    'data/20260113.hairfollicle.part03.tsv.gz',
    'data/20260113.hairfollicle.part04.tsv.gz',
    'data/20260113.hairfollicle.part05.tsv.gz',
    'data/20260113.hairfollicle.part06.tsv.gz',
    'data/20260113.hairfollicle.part07.tsv.gz',
    'data/20260113.hairfollicle.part08.tsv.gz',
    'data/20260113.hairfollicle.part09.tsv.gz',
    'data/20260113.hairfollicle.part10.tsv.gz'
];

// Configuration for downsampling
const TARGET_POINTS_PER_FILE = 100000; // Target number of points to keep per file
const MAX_POINTS = 1000000; // Reduce for performance
    
// Column configuration - users can manually change these lists
let idx_names = ['transformedX', 'transformedZ', 'transformedY']; // Default to transformed coordinates                          
const column_names_categorical = ['Structure', 'HF', 'Sample', 'Group', 'CellType', 'Gene'];
const column_names_continuous = ['Time'];

// Global variables
let scene, camera, renderer, controls;
let points, pointCloud, sphereMesh;
let allData = [];
let visibleIndices = null; // Will be Uint32Array
let colorMap = new Map();
let attributeValues = {}; // Will be dynamically populated
let continuousRanges = {}; // Will store min/max for each continuous variable
let cameraInitialized = false;
let activeFilters = []; // Array of filter objects: { attribute, type: 'categorical'|'time', values/range }
let initialCameraState = { position: null, target: null }; // Store initial camera state for reset
let renderedIndicesMap = null; // Map from instance index to data index for hover detection
let highlightSphere = null; // Sphere to highlight hovered point
let tooltip = null; // Tooltip element
let isShiftPressed = false; // Track SHIFT key state
let raycaster = new THREE.Raycaster(); // For point picking
let mouse = new THREE.Vector2(); // Mouse position for raycasting
let eventListenersInitialized = false; // Track if event listeners have been set up

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
                    const yPanAmount = deltaY * scaledPanSpeed;
                    console.log('[Camera Controls] Panning in Y direction:', yPanAmount.toFixed(4));
                    const yPanVector = new THREE.Vector3(0, yPanAmount, 0);
                    controls.target.add(yPanVector);
                    camera.position.add(yPanVector);
                }
                
                if (Math.abs(deltaX) > 0) {
                    // Pan in x direction (world space)
                    const xPanAmount = deltaX * scaledPanSpeed;
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
                const cameraDistance = camera.position.distanceTo(controls.target);
                const scaledPanSpeed = panSpeed * (cameraDistance * 0.01);
                
                if (Math.abs(deltaY) > 0) {
                    const yPanAmount = deltaY * scaledPanSpeed;
                    const yPanVector = new THREE.Vector3(0, yPanAmount, 0);
                    controls.target.add(yPanVector);
                    camera.position.add(yPanVector);
                }
                
                if (Math.abs(deltaX) > 0) {
                    const xPanAmount = deltaX * scaledPanSpeed;
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
    if (!isShiftPressed || !sphereMesh || !renderedIndicesMap) {
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
    
    // Update raycaster - for instanced meshes, we need to check intersections properly
    const pointSize = parseFloat(document.getElementById('pointSize')?.value || 1);
    raycaster.setFromCamera(mouse, camera);
    
    // Check intersection with instanced mesh
    // Note: Raycaster should automatically handle instanced meshes
    try {
        const intersects = raycaster.intersectObject(sphereMesh, false);
        
        if (intersects.length > 0) {
            const intersection = intersects[0];
            const instanceId = intersection.instanceId;
            
            if (instanceId !== undefined && instanceId !== null && instanceId < renderedIndicesMap.length) {
                const dataIdx = renderedIndicesMap[instanceId];
                if (dataIdx !== undefined && dataIdx < allData.length) {
                    const point = allData[dataIdx];
                    
                    // Show highlight at the actual point position (not intersection point)
                    const pointPosition = new THREE.Vector3(point.x, point.y, point.z);
                    showHighlight(pointPosition, point, event.clientX, event.clientY);
                    
                    // Log for debugging (only occasionally to avoid spam)
                    if (Math.random() < 0.01) { // Log 1% of the time
                        console.log('[Hover] Point highlighted:', {
                            instanceId,
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
    
    // Add coordinates
    tooltipContent += `<div style="margin-bottom: 4px;"><strong>Coordinates:</strong><br>(${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})</div>`;
    
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
        const loadStartTime = Date.now();
        
        // Initialize data structures
        allData = [];
        let headers = null;
        let xIdx, yIdx, zIdx;
        const categoricalIndices = {};
        const continuousIndices = {};
        
        // Process each file individually: load, parse, and downsample
        for (let fileIndex = 0; fileIndex < files.length; fileIndex++) {
            const file = files[fileIndex];
            loadingText.textContent = `Loading and processing file ${fileIndex + 1}/${files.length}...`;
            
            // Load file
            const fileText = await loadGzippedFile(file);
            const fileLines = fileText.split('\n');
            
            // Get headers from first file
            if (fileIndex === 0) {
                headers = fileLines[0].split('\t');
                
                // Find column indices for coordinates
                xIdx = headers.indexOf(idx_names[0]);
                yIdx = headers.indexOf(idx_names[1]);
                zIdx = headers.indexOf(idx_names[2]);
                
                // Validate that coordinate columns were found
                if (xIdx === -1 || yIdx === -1 || zIdx === -1) {
                    const missing = [];
                    if (xIdx === -1) missing.push(idx_names[0]);
                    if (yIdx === -1) missing.push(idx_names[1]);
                    if (zIdx === -1) missing.push(idx_names[2]);
                    console.error(`Error: Coordinate columns not found: ${missing.join(', ')}`);
                    console.error(`Available columns: ${headers.join(', ')}`);
                    throw new Error(`Coordinate columns not found: ${missing.join(', ')}`);
                }
                
                console.log(`Coordinate column indices: ${idx_names[0]}=${xIdx}, ${idx_names[1]}=${yIdx}, ${idx_names[2]}=${zIdx}`);
                
                // Find column indices for categorical and continuous variables
                column_names_categorical.forEach(col => {
                    const idx = headers.indexOf(col);
                    if (idx !== -1) {
                        categoricalIndices[col] = idx;
                        attributeValues[col] = new Set();
                    }
                });
                
                column_names_continuous.forEach(col => {
                    const idx = headers.indexOf(col);
                    if (idx !== -1) {
                        continuousIndices[col] = idx;
                        continuousRanges[col] = { min: Infinity, max: -Infinity };
                    }
                });
            }
            
            // Parse and collect valid points from this file
            const fileData = [];
            const dataStartIdx = 1; // Skip header
            
            for (let i = dataStartIdx; i < fileLines.length; i++) {
                const line = fileLines[i].trim();
                if (!line) continue;
                
                const cols = line.split('\t');
                if (cols.length < headers.length) continue;
                
                // Parse coordinates
                const xStr = (cols[xIdx] || '').trim();
                const yStr = (cols[yIdx] || '').trim();
                const zStr = (cols[zIdx] || '').trim();
                
                const x = parseFloat(xStr);
                const y = parseFloat(yStr);
                const z = parseFloat(zStr);
                
                if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
                
                const point = {
                    x: x,
                    y: y,
                    z: z
                };
                
                // Add categorical attributes
                column_names_categorical.forEach(col => {
                    if (categoricalIndices[col] !== undefined) {
                        const value = cols[categoricalIndices[col]] || '';
                        point[col] = value;
                    }
                });
                
                // Add continuous attributes
                column_names_continuous.forEach(col => {
                    if (continuousIndices[col] !== undefined) {
                        const value = parseFloat(cols[continuousIndices[col]]);
                        const numValue = isNaN(value) ? null : value;
                        point[col] = numValue;
                    }
                });
                
                fileData.push(point);
            }
            
            // Downsample this file's data randomly
            let sampledData = fileData;
            if (fileData.length > TARGET_POINTS_PER_FILE) {
                // Randomly sample TARGET_POINTS_PER_FILE points
                const indices = Array.from({ length: fileData.length }, (_, i) => i);
                // Fisher-Yates shuffle for random sampling
                for (let i = 0; i < TARGET_POINTS_PER_FILE && i < indices.length; i++) {
                    const j = i + Math.floor(Math.random() * (indices.length - i));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                sampledData = indices.slice(0, TARGET_POINTS_PER_FILE).map(idx => fileData[idx]);
                console.log(`Downsampled file ${fileIndex + 1}: ${fileData.length} -> ${sampledData.length} points`);
            }
            
            // Add sampled data to allData
            for (const point of sampledData) {
                allData.push(point);
                
                // Collect unique values for categorical attributes
                column_names_categorical.forEach(col => {
                    if (point[col] && attributeValues[col]) {
                        attributeValues[col].add(point[col]);
                    }
                });
                
                // Track ranges for continuous attributes
                column_names_continuous.forEach(col => {
                    if (point[col] !== null && point[col] !== undefined && continuousRanges[col]) {
                        continuousRanges[col].min = Math.min(continuousRanges[col].min, point[col]);
                        continuousRanges[col].max = Math.max(continuousRanges[col].max, point[col]);
                    }
                });
            }
            
            // Log first few points from first file for debugging
            if (fileIndex === 0 && allData.length > 0) {
                for (let i = 0; i < Math.min(5, allData.length); i++) {
                    const p = allData[i];
                    console.log(`Sample point ${i + 1}: x=${p.x.toFixed(2)}, y=${p.y.toFixed(2)}, z=${p.z.toFixed(2)}`);
                }
            }
            
            // Yield to browser to prevent blocking
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        const loadTime = ((Date.now() - loadStartTime) / 1000).toFixed(1);
        console.log(`Loaded and processed ${files.length} files in ${loadTime}s. Total points: ${allData.length}`);
        
        loadingText.textContent = `Loaded ${allData.length.toLocaleString()} points. Initializing visualization...`;
        document.getElementById('pointCount').textContent = `Total points: ${allData.length.toLocaleString()}`;
        await new Promise(resolve => setTimeout(resolve, 100)); // Brief pause to show final message
        
        // Calculate and log coordinate ranges for verification
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
            console.log(`  ${idx_names[0]}: [${xMin.toFixed(2)}, ${xMax.toFixed(2)}] (span: ${(xMax - xMin).toFixed(2)})`);
            console.log(`  ${idx_names[1]}: [${yMin.toFixed(2)}, ${yMax.toFixed(2)}] (span: ${(yMax - yMin).toFixed(2)})`);
            console.log(`  ${idx_names[2]}: [${zMin.toFixed(2)}, ${zMax.toFixed(2)}] (span: ${(zMax - zMin).toFixed(2)})`);
        }
        
        // Initialize visible indices more efficiently
        visibleIndices = new Uint32Array(allData.length);
        for (let i = 0; i < allData.length; i++) {
            visibleIndices[i] = i;
        }
        
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
        // Set first attribute as default
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
            // Start animation loop (only once)
            animate();
        }
        
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
    
    // Get all unique values for this attribute
    const allValues = new Set();
    if (visibleIndices && visibleIndices.length > 0) {
        for (let i = 0; i < visibleIndices.length; i++) {
            const point = allData[visibleIndices[i]];
            const value = point[colorBy] || '';
            if (value) {
                allValues.add(value);
            }
        }
    }
    
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
    
    // Get unique values from visible data
    const visibleValues = new Set();
    const isContinuous = column_names_continuous.includes(colorBy);
    
    if (visibleIndices && visibleIndices.length > 0) {
        for (let i = 0; i < visibleIndices.length; i++) {
            const point = allData[visibleIndices[i]];
            if (isContinuous) {
                if (point[colorBy] !== null && point[colorBy] !== undefined) {
                    visibleValues.add(point[colorBy]);
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
    const sampleRate = Math.max(0.01, Math.min(1, parseInt(document.getElementById('sampleRate').value) / 100));
    
    // Sample data based on sample rate
    // For spheres, reduce max points slightly for performance
    let indicesToRender = [];
    const visibleCount = visibleIndices.length;
    
    // Calculate target count based on sample rate, but cap at MAX_POINTS for performance
    // When sample rate is 100%, show all points (up to MAX_POINTS limit for performance)
    const targetCount = Math.min(Math.floor(visibleCount * sampleRate), MAX_POINTS);
    
    if (targetCount >= visibleCount) {
        // If target is >= all points (100% sample rate and within limits), use all visible indices
        indicesToRender = Array.from(visibleIndices);
    } else {
        // Randomly sample targetCount points from visible indices
        // Convert Uint32Array to regular array for shuffling
        const allVisibleIndices = Array.from(visibleIndices);
        
        // Fisher-Yates shuffle algorithm for random sampling
        // We only need to shuffle the first targetCount elements
        for (let i = 0; i < targetCount && i < allVisibleIndices.length; i++) {
            // Pick a random index from remaining unshuffled portion
            const j = i + Math.floor(Math.random() * (allVisibleIndices.length - i));
            // Swap
            [allVisibleIndices[i], allVisibleIndices[j]] = [allVisibleIndices[j], allVisibleIndices[i]];
        }
        
        // Take the first targetCount elements (which are now randomly selected)
        indicesToRender = allVisibleIndices.slice(0, targetCount);
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
        const valueForColor = point[colorBy];
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
    
    // Store mapping from instance index to data index for hover detection
    renderedIndicesMap = indicesToRender;
    
    // Update legend
    updateLegend();
    
    // Update info
    const isSampled = indicesToRender.length < visibleIndices.length;
    document.getElementById('visibleCount').textContent = 
        `Visible points: ${indicesToRender.length.toLocaleString()}${isSampled ? ` (sampled from ${visibleIndices.length.toLocaleString()})` : ''}`;
    
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
        
        newSelect.addEventListener('change', (e) => {
            const filterId = e.target.dataset.filterId;
            const attribute = e.target.value;
            
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

// Setup event listeners
function setupEventListeners() {
    // Coordinate system change
    document.getElementById('coordinateSystem').addEventListener('change', async (e) => {
        const selectedSystem = e.target.value;
        console.log(`[Coordinates] Changing to: ${selectedSystem}`);
        
        // Update idx_names based on selection
        if (selectedSystem === 'transformed') {
            idx_names = ['transformedX', 'transformedZ', 'transformedY'];
        } else {
            idx_names = ['x', 'z', 'y'];
        }
        
        console.log(`[Coordinates] Using coordinate columns: ${idx_names.join(', ')}`);
        
        // Reset camera initialization
        cameraInitialized = false;
        
        // Clear filters
        activeFilters = [];
        renderFilters();
        
        // Clear color map to regenerate colors for new coordinates
        colorMap.clear();
        
        // Reload data with new coordinates
        const loadingEl = document.getElementById('loading');
        let loadingText = loadingEl.querySelector('.loading-text');
        if (!loadingText) {
            loadingEl.innerHTML = '<div class="loading-spinner"></div><div class="loading-text">Loading data...</div>';
            loadingText = loadingEl.querySelector('.loading-text');
        }
        loadingEl.style.display = 'flex';
        loadingText.textContent = `Loading data with ${selectedSystem} coordinates...`;
        
        try {
            // Reload data (will skip event listener setup since it's already initialized)
            await loadData();
        } catch (error) {
            console.error('[Coordinates] Error reloading data:', error);
            loadingText.textContent = 'Error loading data. Please check the console.';
            loadingEl.style.background = 'rgba(231, 76, 60, 0.9)';
        }
    });
    
    // Randomize colors button
    const randomizeButton = document.getElementById('randomizeColors');
    if (randomizeButton) {
        randomizeButton.addEventListener('click', () => {
            randomizeColors();
        });
    }
    
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
