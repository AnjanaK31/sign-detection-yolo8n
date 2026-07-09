// Frontend Controller for YOLO-OBB and PaddleOCR Viewer
document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        folderPath: '',
        files: [],
        activeFile: null,
        predictions: null,
        selectedBoxId: null,
        viewMode: 'raw', // 'raw' | 'clean'
        modelType: 'yolo', // 'yolo' | 'paddle'
        showCropsSidebar: false,
        
        // Navigation / Canvas Transform State
        zoom: 1.0,
        panX: 0,
        panY: 0,
        isDragging: false,
        dragStartX: 0,
        dragStartY: 0
    };

    // DOM Elements
    const el = {
        folderPath: document.getElementById('folder-path'),
        loadFolderBtn: document.getElementById('load-folder-btn'),
        fileCountBadge: document.getElementById('file-count-badge'),
        searchInput: document.getElementById('search-input'),
        fileList: document.getElementById('file-list'),
        
        // Model toggle
        toggleModelYolo: document.getElementById('toggle-model-yolo'),
        toggleModelPaddle: document.getElementById('toggle-model-paddle'),
        
        // Workspace
        activeFilenameDisplay: document.getElementById('active-filename-display'),
        toggleRaw: document.getElementById('toggle-raw'),
        toggleClean: document.getElementById('toggle-clean'),
        canvasViewport: document.getElementById('canvas-viewport'),
        canvasWrapper: document.getElementById('canvas-wrapper'),
        displayImage: document.getElementById('display-image'),
        svgOverlay: document.getElementById('svg-overlay'),
        canvasEmptyState: document.getElementById('canvas-empty-state'),
        inferenceLoader: document.getElementById('inference-loader'),
        
        // Zoom controls
        zoomInBtn: document.getElementById('zoom-in-btn'),
        zoomOutBtn: document.getElementById('zoom-out-btn'),
        zoomFitBtn: document.getElementById('zoom-fit-btn'),
        zoomResetBtn: document.getElementById('zoom-reset-btn'),
        zoomPercentage: document.getElementById('zoom-percentage'),
        
        // Tooltip
        tooltip: document.getElementById('canvas-tooltip'),
        
        // Inspector Left/Right
        inspectorImageSummary: document.getElementById('inspector-image-summary'),
        inspectorBoxDetails: document.getElementById('inspector-box-details'),
        backToSummaryBtn: document.getElementById('back-to-summary-btn'),
        
        // Crops Sidebar
        toggleCropsSidebar: document.getElementById('toggle-crops-sidebar'),
        cropsSidebar: document.getElementById('crops-sidebar'),
        cropsCountBadge: document.getElementById('crops-count-badge'),
        cropsListContainer: document.getElementById('crops-list-container'),
        
        // Stats Fields
        summaryTotalBoxes: document.getElementById('summary-total-boxes'),
        summaryAvgConf: document.getElementById('summary-avg-conf'),
        summaryResolution: document.getElementById('summary-resolution'),
        cacheStatus: document.getElementById('cache-status'),
        cacheIndicator: document.getElementById('cache-indicator'),
        cacheDesc: document.getElementById('cache-desc'),
        cacheActionContainer: document.getElementById('cache-action-container'),
        clearCacheBtn: document.getElementById('clear-cache-btn'),
        
        // Box Details Fields
        boxDetailId: document.getElementById('box-detail-id'),
        boxDetailCrop: document.getElementById('box-detail-crop'),
        cropLoader: document.getElementById('crop-loader'),
        boxDetailChar: document.getElementById('box-detail-char'),
        boxDetailClass: document.getElementById('box-detail-class'),
        boxDetailClassConf: document.getElementById('box-detail-class-conf'),
        boxDetailClassBar: document.getElementById('box-detail-class-bar'),
        boxDetailYoloConf: document.getElementById('box-detail-yolo-conf'),
        boxDetailYoloBar: document.getElementById('box-detail-yolo-bar'),
        boxDetailCx: document.getElementById('box-detail-cx'),
        boxDetailCy: document.getElementById('box-detail-cy'),
        boxDetailW: document.getElementById('box-detail-w'),
        boxDetailH: document.getElementById('box-detail-h'),
        boxDetailAngle: document.getElementById('box-detail-angle'),
        boxDetailCornersBody: document.getElementById('box-detail-corners-body'),
        
        // Pipeline stack fields
        stageImgRaw: document.getElementById('stage-img-raw'),
        stageImgThresh: document.getElementById('stage-img-thresh'),
        stageImgDenoised: document.getElementById('stage-img-denoised'),
        stageImgLines: document.getElementById('stage-img-lines'),
        stageImgClean: document.getElementById('stage-img-clean')
    };

    // Constants
    const API_URL = 'http://localhost:8000';

    // Initialize Page
    init();

    function init() {
        // Event Listeners
        el.loadFolderBtn.addEventListener('click', scanDirectory);
        el.searchInput.addEventListener('input', renderFileList);
        el.folderPath.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') scanDirectory();
        });

        // Model Selection Toggles
        el.toggleModelYolo.addEventListener('click', () => setModelType('yolo'));
        el.toggleModelPaddle.addEventListener('click', () => setModelType('paddle'));

        // View Mode Toggles
        el.toggleRaw.addEventListener('click', () => setViewMode('raw'));
        el.toggleClean.addEventListener('click', () => setViewMode('clean'));

        // Toggle Crops Sidebar
        el.toggleCropsSidebar.addEventListener('click', toggleCropsSidebar);

        // Canvas Viewport Panning
        el.canvasViewport.addEventListener('mousedown', startPan);
        window.addEventListener('mousemove', pan);
        window.addEventListener('mouseup', endPan);
        el.canvasViewport.addEventListener('wheel', handleWheel, { passive: false });

        // Zoom Panel
        el.zoomInBtn.addEventListener('click', () => adjustZoom(1.2));
        el.zoomOutBtn.addEventListener('click', () => adjustZoom(1 / 1.2));
        el.zoomResetBtn.addEventListener('click', resetZoom);
        el.zoomFitBtn.addEventListener('click', fitToScreen);

        // Inspector Navigation
        el.backToSummaryBtn.addEventListener('click', showImageSummary);
        el.clearCacheBtn.addEventListener('click', clearCurrentCache);

        // Auto load initial folder if set
        if (el.folderPath.value) {
            scanDirectory();
        }
    }

    // API Calls
    async function scanDirectory() {
        const pathVal = el.folderPath.value.trim();
        if (!pathVal) return;
        
        el.loadFolderBtn.disabled = true;
        el.loadFolderBtn.classList.add('loading');
        
        try {
            const res = await fetch(`${API_URL}/api/images?dir=${encodeURIComponent(pathVal)}`);
            const data = await res.json();
            
            if (data.error) {
                alert(`Error scanning folder: ${data.error}`);
                state.files = [];
            } else {
                state.folderPath = pathVal;
                state.files = data.images || [];
            }
        } catch (err) {
            console.error(err);
            alert('Failed to connect to backend server. Make sure yolo_viewer.py is running on port 8000.');
            state.files = [];
        } finally {
            el.loadFolderBtn.disabled = false;
            el.loadFolderBtn.classList.remove('loading');
            renderFileList();
            
            // Clear current workspace since folder changed
            clearWorkspace();
        }
    }

    function setModelType(model) {
        if (state.modelType === model) return;
        state.modelType = model;
        
        if (model === 'yolo') {
            el.toggleModelYolo.classList.add('active');
            el.toggleModelPaddle.classList.remove('active');
            el.inferenceLoader.querySelector('.loader-title').textContent = 'Running YOLOv8-OBB Inference';
        } else {
            el.toggleModelYolo.classList.remove('active');
            el.toggleModelPaddle.classList.add('active');
            el.inferenceLoader.querySelector('.loader-title').textContent = 'Running PaddleOCR Inference';
        }
        
        // Re-render file list to reflect correct status badges
        renderFileList();
        
        // If an active image is selected, re-request prediction using the newly active model
        if (state.activeFile) {
            const active = state.activeFile;
            state.activeFile = null; // force reload
            state.predictions = null;
            selectImage(active);
        }
    }

    function renderFileList() {
        const query = el.searchInput.value.toLowerCase().trim();
        const filtered = state.files.filter(f => f.filename.toLowerCase().includes(query));
        
        el.fileCountBadge.textContent = filtered.length;
        el.fileList.innerHTML = '';
        
        if (filtered.length === 0) {
            el.fileList.innerHTML = `
                <div class="empty-state">
                    <p>${state.folderPath ? 'No matching images found.' : 'No scanned folder.'}</p>
                </div>
            `;
            return;
        }

        filtered.forEach(file => {
            const li = document.createElement('li');
            li.className = `file-item ${state.activeFile && state.activeFile.filename === file.filename ? 'active' : ''}`;
            
            // Check prediction flag based on currently active model
            const isPredicted = state.modelType === 'yolo' ? file.predicted_yolo : file.predicted_paddle;
            const predictedClass = isPredicted ? 'predicted' : 'pending';
            const predictedText = isPredicted ? 'Predicted' : 'Pending';
            
            li.innerHTML = `
                <div class="file-info">
                    <span class="file-name" title="${file.filename}">${file.filename}</span>
                    <span class="file-meta">Image file</span>
                </div>
                <span class="status-badge ${predictedClass}">${predictedText}</span>
            `;
            
            li.addEventListener('click', () => selectImage(file));
            el.fileList.appendChild(li);
        });
    }
    function clearWorkspace() {
        state.activeFile = null;
        state.predictions = null;
        state.selectedBoxId = null;
        
        if (window.activePaddleSocket) {
            window.activePaddleSocket.close();
            window.activePaddleSocket = null;
        }
        
        el.activeFilenameDisplay.textContent = 'No image selected';
        el.displayImage.src = '';
        el.displayImage.classList.add('hidden');
        el.svgOverlay.classList.add('hidden');
        el.canvasEmptyState.classList.remove('hidden');
        el.inferenceLoader.classList.add('hidden');
        
        el.cropsListContainer.innerHTML = `
            <div class="empty-state">
                <p>No active crops. Select an image and run predictions.</p>
            </div>
        `;
        el.cropsCountBadge.textContent = '0';
        
        showImageSummary();
        displayPerformanceTimes(0, 0, 0);
    }

    async function selectImage(file) {
        if (window.activePaddleSocket) {
            window.activePaddleSocket.close();
            window.activePaddleSocket = null;
        }

        if (state.activeFile && state.activeFile.filename === file.filename && state.predictions) {
            // Already active, don't re-fetch
            return;
        }
        
        state.activeFile = file;
        state.selectedBoxId = null;
        
        // Highlight in list
        const items = el.fileList.querySelectorAll('.file-item');
        items.forEach(item => {
            const nameEl = item.querySelector('.file-name');
            if (nameEl && nameEl.textContent === file.filename) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Set visual loading states
        el.activeFilenameDisplay.textContent = file.filename;
        el.canvasEmptyState.classList.add('hidden');
        el.displayImage.classList.add('hidden');
        el.svgOverlay.classList.add('hidden');
        
        // Show loader if not predicted
        const isPredicted = state.modelType === 'yolo' ? file.predicted_yolo : file.predicted_paddle;
        if (!isPredicted) {
            el.inferenceLoader.classList.remove('hidden');
        }
        
        showImageSummary();
        updateCacheStatus(isPredicted ? 'loading' : 'predicting');

        // PaddleOCR live predictions go through WebSockets
        if (state.modelType === 'paddle') {
            runPaddleWebSocket(file);
            return;
        }

        try {
            // Load prediction API (YOLO path)
            const body = {
                dir_path: state.folderPath,
                filename: file.filename,
                model_type: state.modelType
            };
            
            const res = await fetch(`${API_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            
            const data = await res.json();
            
            if (data.error) {
                alert(`Error running prediction: ${data.error}`);
                clearWorkspace();
                return;
            }
            
            state.predictions = data;
            
            // Mark file as predicted in local state
            file.predicted_yolo = true;
            renderFileList();
            
            // Set image source based on active viewMode
            renderActiveImage();
            
            // Display overlay
            drawSvgOverlay();
            
            // Refresh stats
            renderImageStats();
            
            renderCropsSidebar();
            
            // Fit to viewport after rendering
            setTimeout(fitToScreen, 100);

        } catch (err) {
            console.error(err);
            alert('Error running model inference.');
            clearWorkspace();
        } finally {
            el.inferenceLoader.classList.add('hidden');
        }
    }

    function runPaddleWebSocket(file) {
        const wsUrl = `${API_URL.replace(/^http/, 'ws')}/api/ws/paddle`;
        
        let socket;
        try {
            socket = new WebSocket(wsUrl);
        } catch (e) {
            console.error('Failed to create WebSocket:', e);
            alert('Failed to connect to PaddleOCR WebSocket.');
            clearWorkspace();
            return;
        }
        
        if (window.activePaddleSocket) {
            window.activePaddleSocket.close();
        }
        window.activePaddleSocket = socket;
        
        socket.onopen = () => {
            console.log('WS: Connected to PaddleOCR service');
            socket.send(JSON.stringify({
                dir_path: state.folderPath,
                filename: file.filename
            }));
        };
        
        socket.onmessage = (e) => {
            const data = JSON.parse(e.data);
            
            if (data.error) {
                alert(`Error during PaddleOCR inference: ${data.error}`);
                clearWorkspace();
                socket.close();
                return;
            }
            
            if (data.type === 'detection') {
                console.log('WS: Received detection bounding boxes:', data.boxes.length);
                state.predictions = {
                    filepath: `${state.folderPath}/${file.filename}`,
                    width: data.width,
                    height: data.height,
                    boxes: data.boxes
                };
                
                renderActiveImage();
                drawSvgOverlay();
                renderImageStats();
                renderCropsSidebar();
                el.inferenceLoader.classList.add('hidden');
                updateCacheStatus('predicting');
                setTimeout(fitToScreen, 100);
            }
            else if (data.type === 'recognition') {
                console.log(`WS: Received recognition for box ${data.id}: "${data.char_display}"`);
                if (state.predictions && state.predictions.boxes) {
                    const box = state.predictions.boxes.find(b => b.id === data.id);
                    if (box) {
                        box.char_display = data.char_display;
                        box.class_confidence = data.class_confidence;
                        box.confidence = data.class_confidence;
                        box.completed = true;
                        
                        updateSvgBoxState(data.id, data.char_display, data.class_confidence);
                        renderImageStats();
                    }
                }
            }
            else if (data.type === 'complete') {
                console.log('WS: Prediction completed');
                file.predicted_paddle = true;
                renderFileList();
                updateCacheStatus('predicted');
                el.inferenceLoader.classList.add('hidden');
                
                // Show performance metrics in UI
                displayPerformanceTimes(data.det_time || 0.0, data.rec_time || 0.0, data.total_time || 0.0);
                
                window.activePaddleSocket = null;
            }
        };
        
        socket.onerror = (err) => {
            console.error('WS error:', err);
            alert('PaddleOCR WebSocket connection error.');
            clearWorkspace();
        };
        
        socket.onclose = () => {
            console.log('WS: Connection closed');
            el.inferenceLoader.classList.add('hidden');
            if (window.activePaddleSocket === socket) {
                window.activePaddleSocket = null;
            }
        };
    }
    
    function updateSvgBoxState(id, text, conf) {
        const poly = el.svgOverlay.querySelector(`polygon[data-id="${id}"]`);
        if (poly) {
            poly.classList.remove('pending');
            poly.classList.add('completed');
        }
        
        updateCropsSidebarCardText(id, text);
        
        if (state.selectedBoxId === id) {
            showBoxDetail(id);
        }
    }

    function renderActiveImage() {
        if (!state.predictions) return;
        
        const modeQuery = state.viewMode === 'clean' ? '&cleaned=true' : '';
        const imgUrl = `${API_URL}/api/image-file?filepath=${encodeURIComponent(state.predictions.filepath)}${modeQuery}`;
        
        el.displayImage.onload = () => {
            el.displayImage.classList.remove('hidden');
            el.svgOverlay.classList.remove('hidden');
            
            // Align SVG viewport attributes
            el.svgOverlay.setAttribute('viewBox', `0 0 ${state.predictions.width} ${state.predictions.height}`);
            
            // Reset wrapper size based on aspect ratio
            el.canvasWrapper.style.width = `${state.predictions.width}px`;
            el.canvasWrapper.style.height = `${state.predictions.height}px`;
        };
        
        el.displayImage.src = imgUrl;
    }

    function setViewMode(mode) {
        if (state.viewMode === mode) return;
        state.viewMode = mode;
        
        if (mode === 'raw') {
            el.toggleRaw.classList.add('active');
            el.toggleClean.classList.remove('active');
        } else {
            el.toggleRaw.classList.remove('active');
            el.toggleClean.classList.add('active');
        }
        
        if (state.predictions) {
            renderActiveImage();
        }
    }

    // SVG Rendering
    function drawSvgOverlay() {
        el.svgOverlay.innerHTML = '';
        if (!state.predictions || !state.predictions.boxes) return;
        
        state.predictions.boxes.forEach(box => {
            const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            const ptsStr = box.corners.map(p => `${p[0]},${p[1]}`).join(' ');
            
            poly.setAttribute('points', ptsStr);
            let stateClass = '';
            if (state.modelType === 'paddle') {
                stateClass = box.completed ? 'completed' : 'pending';
            }
            poly.setAttribute('class', `detection-poly ${stateClass} ${state.selectedBoxId === box.id ? 'selected' : ''}`);
            poly.setAttribute('data-id', box.id);
            
            // Hover Events
            poly.addEventListener('mouseenter', (e) => showTooltip(e, box));
            poly.addEventListener('mousemove', moveTooltip);
            poly.addEventListener('mouseleave', hideTooltip);
            
            // Click Events
            poly.addEventListener('click', (e) => {
                e.stopPropagation(); // Avoid deselection when clicking canvas background
                selectBox(box.id);
            });
            
            el.svgOverlay.appendChild(poly);
        });
        
        // Clicking background clears selection
        el.svgOverlay.addEventListener('click', () => {
            selectBox(null);
        });
    }

    // Tooltip Handling
    function showTooltip(e, box) {
        el.tooltip.querySelector('.tooltip-index').textContent = `BOX #${box.id + 1}`;
        
        if (state.modelType === 'yolo') {
            el.tooltip.querySelector('.tooltip-char').textContent = `Symbol: '${box.char_display}'`;
            el.tooltip.querySelector('.tooltip-conf').textContent = `Conf: ${(box.class_confidence * 100).toFixed(1)}%`;
        } else {
            el.tooltip.querySelector('.tooltip-char').textContent = `OCR: "${box.char_display}"`;
            el.tooltip.querySelector('.tooltip-conf').textContent = `Det Score: ${(box.confidence * 100).toFixed(1)}%`;
        }
        
        el.tooltip.querySelector('.tooltip-angle').textContent = `Angle: ${box.angle > 0 ? '+' : ''}${box.angle.toFixed(1)}°`;
        
        el.tooltip.classList.remove('hidden');
        moveTooltip(e);
    }

    function moveTooltip(e) {
        const tooltipW = el.tooltip.clientWidth;
        const tooltipH = el.tooltip.clientHeight;
        
        // Offset from cursor
        const offset = 15;
        
        let x = e.clientX + offset;
        let y = e.clientY + offset;
        
        // Boundary checks
        if (x + tooltipW > window.innerWidth) {
            x = e.clientX - tooltipW - offset;
        }
        if (y + tooltipH > window.innerHeight) {
            y = e.clientY - tooltipH - offset;
        }
        
        el.tooltip.style.left = `${x}px`;
        el.tooltip.style.top = `${y}px`;
    }

    function hideTooltip() {
        el.tooltip.classList.add('hidden');
    }

    // Box Selection
    function selectBox(boxId) {
        state.selectedBoxId = boxId;
        
        // Update SVG styles
        const polys = el.svgOverlay.querySelectorAll('.detection-poly');
        polys.forEach(p => {
            const id = parseInt(p.getAttribute('data-id'));
            if (id === boxId) {
                p.classList.add('selected');
            } else {
                p.classList.remove('selected');
            }
        });
        
        // Update Crops Sidebar active card styling
        const cards = el.cropsListContainer.querySelectorAll('.crop-thumbnail-card');
        cards.forEach(card => {
            const id = parseInt(card.getAttribute('data-box-id'));
            if (id === boxId) {
                card.classList.add('active');
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                card.classList.remove('active');
            }
        });
        
        if (boxId === null) {
            showImageSummary();
        } else {
            showBoxDetail(boxId);
        }
    }

    // Side Panels Rendering
    function showImageSummary() {
        el.inspectorImageSummary.classList.remove('hidden');
        el.inspectorBoxDetails.classList.add('hidden');
        clearPipelineImages();
    }

    function updateCacheStatus(status) {
        el.cacheIndicator.className = 'cache-indicator-dot';
        const modelLabel = state.modelType === 'yolo' ? 'YOLOv8' : 'PaddleOCR';
        
        if (status === 'predicted') {
            el.cacheIndicator.classList.add('predicted');
            el.cacheStatus.textContent = 'Predicted & Cached';
            el.cacheDesc.textContent = `This file has ${modelLabel} bounding boxes cached on disk. It will load instantly.`;
            el.cacheActionContainer.classList.remove('hidden');
        } else if (status === 'predicting') {
            el.cacheIndicator.classList.add('unpredicted');
            el.cacheStatus.textContent = 'Running Model...';
            el.cacheDesc.textContent = `${modelLabel} is running. Drawing overlays shortly...`;
            el.cacheActionContainer.classList.add('hidden');
        } else if (status === 'loading') {
            el.cacheIndicator.classList.add('predicted');
            el.cacheStatus.textContent = 'Loading Cache...';
            el.cacheDesc.textContent = `Reading cached ${modelLabel} coordinate file from predictions/ directory...`;
            el.cacheActionContainer.classList.add('hidden');
        } else {
            el.cacheIndicator.classList.add('unpredicted');
            el.cacheStatus.textContent = 'Unpredicted';
            el.cacheDesc.textContent = 'Coordinates have not been generated yet. Click select to trigger model.';
            el.cacheActionContainer.classList.add('hidden');
        }
    }

    function renderImageStats() {
        if (!state.predictions) return;
        
        const boxesCount = state.predictions.boxes.length;
        el.summaryTotalBoxes.textContent = boxesCount;
        el.summaryResolution.textContent = `${state.predictions.width} x ${state.predictions.height} px`;
        
        if (boxesCount > 0) {
            const sumConf = state.predictions.boxes.reduce((acc, b) => acc + b.class_confidence, 0);
            el.summaryAvgConf.textContent = `${((sumConf / boxesCount) * 100).toFixed(1)}%`;
        } else {
            el.summaryAvgConf.textContent = '0%';
        }
        
        updateCacheStatus('predicted');
        
        if (state.modelType === 'paddle' && state.predictions) {
            displayPerformanceTimes(
                state.predictions.det_time || 0.0,
                state.predictions.rec_time || 0.0,
                state.predictions.total_time || 0.0
            );
        } else {
            displayPerformanceTimes(0, 0, 0);
        }
    }

    async function showBoxDetail(boxId) {
        if (!state.predictions || !state.predictions.boxes) return;
        
        const box = state.predictions.boxes.find(b => b.id === boxId);
        if (!box) return;

        el.inspectorImageSummary.classList.add('hidden');
        el.inspectorBoxDetails.classList.remove('hidden');
        
        // Clear previous pipeline images before loading new ones
        clearPipelineImages();
        
        // Text info
        el.boxDetailId.textContent = box.id + 1;
        el.boxDetailChar.textContent = box.char_display;
        el.boxDetailClass.textContent = box.class_name;
        
        // Progress Bars
        const classConfPercent = (box.class_confidence * 100).toFixed(1);
        el.boxDetailClassConf.textContent = `${classConfPercent}%`;
        el.boxDetailClassBar.style.width = `${classConfPercent}%`;
        
        const yoloConfPercent = (box.confidence * 100).toFixed(1);
        el.boxDetailYoloConf.textContent = `${yoloConfPercent}%`;
        el.boxDetailYoloBar.style.width = `${yoloConfPercent}%`;
        
        // If PaddleOCR: Update labels dynamically
        if (state.modelType === 'paddle') {
            document.querySelector('.metric-row.border-b .metric-label').textContent = 'OCR Recognized Text';
            document.querySelectorAll('.metric-row .metric-label')[1].textContent = 'Source Method';
            el.boxDetailClass.textContent = 'paddleocr';
            document.querySelectorAll('.metric-row .metric-label')[2].textContent = 'PaddleOCR Match Score';
            document.querySelectorAll('.metric-row .metric-label')[3].textContent = 'PaddleOCR Detection Score';
        } else {
            document.querySelector('.metric-row.border-b .metric-label').textContent = 'Predicted Symbol';
            document.querySelectorAll('.metric-row .metric-label')[1].textContent = 'Classifier Name';
            document.querySelectorAll('.metric-row .metric-label')[2].textContent = 'Classifier Confidence';
            document.querySelectorAll('.metric-row .metric-label')[3].textContent = 'YOLO Location Confidence';
        }
        
        // Coordinates
        el.boxDetailCx.textContent = `${box.cx.toFixed(1)}px`;
        el.boxDetailCy.textContent = `${box.cy.toFixed(1)}px`;
        el.boxDetailW.textContent = `${box.w.toFixed(1)}px`;
        el.boxDetailH.textContent = `${box.h.toFixed(1)}px`;
        el.boxDetailAngle.textContent = `${box.angle > 0 ? '+' : ''}${box.angle.toFixed(1)}°`;
        
        // Corners table
        el.boxDetailCornersBody.innerHTML = '';
        const cornerLabels = ['Top-Left (TL)', 'Top-Right (TR)', 'Bottom-Right (BR)', 'Bottom-Left (BL)'];
        box.corners.forEach((pt, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${cornerLabels[i]}</td>
                <td>${pt[0].toFixed(1)}</td>
                <td>${pt[1].toFixed(1)}</td>
            `;
            el.boxDetailCornersBody.appendChild(tr);
        });

        // Load Rectified Crop image
        el.boxDetailCrop.src = '';
        el.cropLoader.classList.remove('hidden');
        
        try {
            // For PaddleOCR, force running the MobileNet character classifier on the crop to predict the symbol overlay
            const forceClassify = state.modelType === 'paddle' ? '&force_classify=true' : '';
            const res = await fetch(`${API_URL}/api/crop?filepath=${encodeURIComponent(state.predictions.filepath)}&cx=${box.cx}&cy=${box.cy}&w=${box.w}&h=${box.h}&angle=${box.angle}${forceClassify}`);
            const data = await res.json();
            
            if (data.error) {
                console.error(data.error);
                el.boxDetailCrop.src = '';
            } else {
                el.boxDetailCrop.src = data.crop;
                
                // Load stages images into stack
                if (data.stages) {
                    el.stageImgRaw.src = data.stages.raw || '';
                    el.stageImgThresh.src = data.stages.thresh || '';
                    el.stageImgDenoised.src = data.stages.denoised || '';
                    el.stageImgLines.src = data.stages.lines || '';
                    el.stageImgClean.src = data.stages.clean || '';
                }
                
                // If PaddleOCR ran dynamic classification, update metric card details
                if (state.modelType === 'paddle' && data.char_display && data.class_name) {
                    el.boxDetailChar.textContent = `${box.char_display} [Classified: ${data.char_display}]`;
                    el.boxDetailClass.textContent = `paddleocr + ${data.class_name}`;
                    const clsPercent = (data.class_confidence * 100).toFixed(1);
                    el.boxDetailClassConf.textContent = `${clsPercent}%`;
                    el.boxDetailClassBar.style.width = `${clsPercent}%`;
                }
            }
        } catch (err) {
            console.error('Error fetching crop:', err);
        } finally {
            el.cropLoader.classList.add('hidden');
        }
    }

    // Cache Clearing Logic
    async function clearCurrentCache() {
        if (!state.activeFile) return;
        
        const modelLabel = state.modelType === 'yolo' ? 'YOLOv8' : 'PaddleOCR';
        const confirmClear = confirm(`Are you sure you want to clear predictions cache for ${state.activeFile.filename} (${modelLabel})? This will run the model again to regenerate boxes.`);
        if (!confirmClear) return;
        
        updateCacheStatus('predicting');
        el.inferenceLoader.classList.remove('hidden');
        
        try {
            const body = {
                dir_path: state.folderPath,
                filename: state.activeFile.filename,
                model_type: state.modelType
            };
            
            const res = await fetch(`${API_URL}/api/clear-cache`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            
            if (data.error) {
                alert(`Error clearing cache: ${data.error}`);
                updateCacheStatus('predicted');
                el.inferenceLoader.classList.add('hidden');
            } else {
                // Clear prediction flag for active file
                if (state.modelType === 'yolo') {
                    state.activeFile.predicted_yolo = false;
                } else {
                    state.activeFile.predicted_paddle = false;
                }
                
                // Immediately request prediction to re-generate cache
                const active = state.activeFile;
                state.activeFile = null; // reset to force re-render
                selectImage(active);
            }
        } catch (err) {
            console.error(err);
            alert('Failed to clear cache.');
            updateCacheStatus('predicted');
            el.inferenceLoader.classList.add('hidden');
        }
    }

    // Zoom & Pan Navigation Handlers
    // Drag/Scroll panning, Mouse wheel zoom, fit-screen, reset etc.
    function startPan(e) {
        if (e.target.closest('.zoom-controls') || e.target.closest('.workspace-toolbar')) return;
        
        state.isDragging = true;
        state.dragStartX = e.clientX - state.panX;
        state.dragStartY = e.clientY - state.panY;
        el.canvasViewport.style.cursor = 'grabbing';
    }

    function pan(e) {
        if (!state.isDragging) return;
        
        state.panX = e.clientX - state.dragStartX;
        state.panY = e.clientY - state.dragStartY;
        applyTransform();
    }

    function endPan() {
        state.isDragging = false;
        el.canvasViewport.style.cursor = 'grab';
    }

    function handleWheel(e) {
        e.preventDefault();
        
        const zoomFactor = 1.15;
        const oldZoom = state.zoom;
        
        // Zoom in or out
        if (e.deltaY < 0) {
            state.zoom = Math.min(state.zoom * zoomFactor, 12);
        } else {
            state.zoom = Math.max(state.zoom / zoomFactor, 0.1);
        }

        // Adjust pans to zoom into mouse cursor position
        const rect = el.canvasViewport.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        state.panX = mouseX - (mouseX - state.panX) * (state.zoom / oldZoom);
        state.panY = mouseY - (mouseY - state.panY) * (state.zoom / oldZoom);
        
        applyTransform();
    }

    function adjustZoom(factor) {
        const oldZoom = state.zoom;
        state.zoom = Math.min(Math.max(state.zoom * factor, 0.1), 12);
        
        // Zoom from viewport center
        const cx = el.canvasViewport.clientWidth / 2;
        const cy = el.canvasViewport.clientHeight / 2;
        
        state.panX = cx - (cx - state.panX) * (state.zoom / oldZoom);
        state.panY = cy - (cy - state.panY) * (state.zoom / oldZoom);
        
        applyTransform();
    }

    function resetZoom() {
        state.zoom = 1.0;
        state.panX = 0;
        state.panY = 0;
        applyTransform();
    }

    function fitToScreen() {
        if (!state.predictions) return;
        
        const vpW = el.canvasViewport.clientWidth;
        const vpH = el.canvasViewport.clientHeight;
        const imgW = state.predictions.width;
        const imgH = state.predictions.height;
        
        if (!imgW || !imgH) return;
        
        // Compute fits with a 5% border buffer
        const scaleX = vpW / imgW;
        const scaleY = vpH / imgH;
        const fitScale = Math.min(scaleX, scaleY) * 0.92;
        
        state.zoom = fitScale;
        
        // Center image
        state.panX = (vpW - imgW * fitScale) / 2;
        state.panY = (vpH - imgH * fitScale) / 2;
        
        applyTransform();
    }

    function displayPerformanceTimes(det, rec, total) {
        const container = document.getElementById('inference-times-container');
        if (!container) return;
        
        if (state.modelType === 'paddle' && (det > 0 || rec > 0 || total > 0)) {
            document.getElementById('paddle-det-time').textContent = `${det.toFixed(2)}s`;
            document.getElementById('paddle-rec-time').textContent = `${rec.toFixed(2)}s`;
            document.getElementById('paddle-total-time').textContent = `${total.toFixed(2)}s`;
            container.classList.remove('hidden');
        } else {
            container.classList.add('hidden');
        }
    }

    function applyTransform() {
        el.canvasWrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
        el.zoomPercentage.textContent = `${Math.round(state.zoom * 100)}%`;
    }

    // Toggle Crops Overview Sidebar
    function toggleCropsSidebar() {
        state.showCropsSidebar = !state.showCropsSidebar;
        if (state.showCropsSidebar) {
            el.cropsSidebar.classList.remove('hidden');
            el.toggleCropsSidebar.classList.add('active');
            el.toggleCropsSidebar.querySelector('span').textContent = 'Hide Crops';
        } else {
            el.cropsSidebar.classList.add('hidden');
            el.toggleCropsSidebar.classList.remove('active');
            el.toggleCropsSidebar.querySelector('span').textContent = 'Show Crops';
        }
        setTimeout(fitToScreen, 100);
    }

    // Render the Crops Sidebar Thumbnail Cards
    function renderCropsSidebar() {
        el.cropsListContainer.innerHTML = '';
        if (!state.predictions || !state.predictions.boxes) {
            el.cropsCountBadge.textContent = '0';
            el.cropsListContainer.innerHTML = `
                <div class="empty-state">
                    <p>No active crops. Select an image and run predictions.</p>
                </div>
            `;
            return;
        }
        
        const boxes = state.predictions.boxes;
        el.cropsCountBadge.textContent = boxes.length;
        
        if (boxes.length === 0) {
            el.cropsListContainer.innerHTML = `
                <div class="empty-state">
                    <p>No symbols detected on this page.</p>
                </div>
            `;
            return;
        }
        
        boxes.forEach(box => {
            const card = document.createElement('div');
            card.className = `crop-thumbnail-card ${state.selectedBoxId === box.id ? 'active' : ''}`;
            card.setAttribute('data-box-id', box.id);
            
            // Build the card markup
            card.innerHTML = `
                <div class="crop-thumb-img-wrapper">
                    <div class="spinner-small" id="thumb-loader-${box.id}"></div>
                    <img id="thumb-img-${box.id}" class="crop-thumb-img hidden" src="" alt="Crop ${box.id + 1}">
                </div>
                <div class="crop-thumb-meta">
                    <span class="crop-thumb-id">#${box.id + 1}</span>
                    <span class="crop-thumb-char" id="thumb-char-${box.id}" title="${box.char_display || ''}">${box.char_display || ''}</span>
                </div>
            `;
            
            card.addEventListener('click', (e) => {
                e.stopPropagation();
                selectBox(box.id);
            });
            
            el.cropsListContainer.appendChild(card);
            
            // Asynchronously load thumbnail image
            loadThumbnailImage(box);
        });
    }

    // Load dynamic cropped thumbnail via the fast path API
    async function loadThumbnailImage(box) {
        try {
            const res = await fetch(`${API_URL}/api/crop?filepath=${encodeURIComponent(state.predictions.filepath)}&cx=${box.cx}&cy=${box.cy}&w=${box.w}&h=${box.h}&angle=${box.angle}&thumb_only=true`);
            const data = await res.json();
            
            if (data.crop) {
                const img = document.getElementById(`thumb-img-${box.id}`);
                const loader = document.getElementById(`thumb-loader-${box.id}`);
                if (img && loader) {
                    img.src = data.crop;
                    img.classList.remove('hidden');
                    loader.classList.add('hidden');
                }
            }
        } catch (err) {
            console.error(`Error loading thumbnail for box ${box.id}:`, err);
        }
    }

    // Real-time card character text updating (WebSocket updates)
    function updateCropsSidebarCardText(id, text) {
        const charBadge = document.getElementById(`thumb-char-${id}`);
        if (charBadge) {
            charBadge.textContent = text;
            charBadge.title = text;
        }
    }

    // Clean pipeline stage images inside inspector
    function clearPipelineImages() {
        el.stageImgRaw.src = '';
        el.stageImgThresh.src = '';
        el.stageImgDenoised.src = '';
        el.stageImgLines.src = '';
        el.stageImgClean.src = '';
    }
});
