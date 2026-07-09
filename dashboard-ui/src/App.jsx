import { useState, useEffect, useRef } from 'react';
import { Play, SkipForward, FileText, CheckCircle2, Loader2, Maximize2, LayoutGrid, Image as ImageIcon, Activity, Crosshair, Target, AlertTriangle, ChevronRight, Eye, RefreshCw, Check, Trash2, Sliders, ArrowRight, ArrowLeft, ChevronUp, ChevronDown, ArrowUpDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [mode, setMode] = useState('pipeline'); // 'pipeline' | 'validate' | 'report'
  
  // Pipeline State
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [isPdfLoading, setIsPdfLoading] = useState(false);
  const [pagesList, setPagesList] = useState([]); // [{page_num, image, width, height}]
  const [activePageNum, setActivePageNum] = useState(1);
  
  // Interactive Step Control
  const [yoloStatus, setYoloStatus] = useState('idle'); // 'idle' | 'running' | 'done'
  const [yoloBoxes, setYoloBoxes] = useState([]);
  const [cropsStatus, setCropsStatus] = useState('idle'); // 'idle' | 'running' | 'done'
  const [detections, setDetections] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState(null); // Selected detection for detail journey modal
  
  // Search & Filter States for Crops Grid
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [minConfidence, setMinConfidence] = useState(0);
  const [magnifierCoords, setMagnifierCoords] = useState(null);

  // Validation State
  const [valImages, setValImages] = useState([]);
  const [selectedValImage, setSelectedValImage] = useState(null);
  const [valStep, setValStep] = useState('idle');
  const [gtBoxes, setGtBoxes] = useState([]);
  const [valStats, setValStats] = useState(null);
  const [valImageState, setValImageState] = useState({ image: null, w: 0, h: 0, boxes: [] });

  // Batch Report Viewer State
  const [reportsList, setReportsList] = useState([]);
  const [selectedReportName, setSelectedReportName] = useState(null);
  const [activeReportData, setActiveReportData] = useState(null);
  const [activeReportImage, setActiveReportImage] = useState('');
  const [isReportLoading, setIsReportLoading] = useState(false);
  const [reportSearchText, setReportSearchText] = useState('');
  const [reportStatusFilter, setReportStatusFilter] = useState('all');
  const [reportMinConfidence, setReportMinConfidence] = useState(0);
  const [reportSortField, setReportSortField] = useState('id');
  const [reportSortAsc, setReportSortAsc] = useState(true);
  const [hoveredReportBox, setHoveredReportBox] = useState(null);
  const [reportImageSize, setReportImageSize] = useState({ w: 1000, h: 1000 });
  const [selectedReportRowId, setSelectedReportRowId] = useState(null);
  const [isLoadingCrops, setIsLoadingCrops] = useState(false);

  // Shared State
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredBox, setHoveredBox] = useState(null);
  
  // Interactive Zoom State
  const containerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  
  const ws = useRef(null);

  // Fetch file lists
  useEffect(() => {
    if (mode === 'pipeline') {
      fetch('http://localhost:8000/pdfs')
        .then(res => res.json())
        .then(data => { if(data.pdfs) setPdfs(data.pdfs); })
        .catch(err => console.error(err));
    } else if (mode === 'validate') {
      fetch('http://localhost:8000/val-images')
        .then(res => res.json())
        .then(data => { if(data.images) setValImages(data.images); })
        .catch(err => console.error(err));
    } else if (mode === 'report') {
      fetch('http://localhost:8000/reports')
        .then(res => res.json())
        .then(data => { if(data.reports) setReportsList(data.reports); })
        .catch(err => console.error(err));
    }
  }, [mode]);

  // Reset page states on active page change
  useEffect(() => {
    setYoloStatus('idle');
    setYoloBoxes([]);
    setCropsStatus('idle');
    setDetections([]);
    setZoom(1);
    setSelectedCrop(null);
    setSearchText('');
    setStatusFilter('all');
    setMinConfidence(0);
    setMagnifierCoords(null);
  }, [activePageNum]);

  // Reset report states when report changes
  useEffect(() => {
    setReportSearchText('');
    setReportStatusFilter('all');
    setReportMinConfidence(0);
    setReportSortField('id');
    setReportSortAsc(true);
    setHoveredReportBox(null);
    setSelectedReportRowId(null);
    setZoom(1);
  }, [selectedReportName]);

  const handleSelectReport = (reportName) => {
    setSelectedReportName(reportName);
    setIsReportLoading(true);
    setActiveReportData(null);
    setActiveReportImage('');
    setError(null);
    setSelectedCrop(null);
    setSelectedReportRowId(null);

    fetch(`http://localhost:8000/report/load?name=${reportName}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
        } else {
          setActiveReportData(data.report);
          setActiveReportImage(data.image);
        }
        setIsReportLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setIsReportLoading(false);
      });
  };

  const handleSelectReportCrop = (det) => {
    setSelectedCrop({
      id: det.id - 1, // normalize to 0-based for detail journey modal title (renders as id + 1)
      yolo_conf: det.yolo_conf,
      pred_class: det.prediction,
      pred_char: det.prediction,
      class_confidence: det.class_conf,
      rotation_degrees: det.pred_angle,
      best_iou: det.iou,
      is_class_correct: det.is_class_correct,
      is_orient_correct: det.is_orient_correct,
      gt_expected_char: det.gt_expected,
      gt_angle: det.gt_angle,
      crops: null // to load
    });

    setIsLoadingCrops(true);
    const imgName = activeReportData ? activeReportData.image : '';

    fetch(`http://localhost:8000/report/crop?image_name=${imgName}&cx=${det.cx}&cy=${det.cy}&w=${det.w}&h=${det.h}&angle=${det.pred_angle}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
        } else if (data.crops) {
          setSelectedCrop(prev => {
            if (!prev || prev.id !== det.id - 1) return prev;
            return {
              ...prev,
              crops: data.crops
            };
          });
        }
        setIsLoadingCrops(false);
      })
      .catch(err => {
        setError(err.message);
        setIsLoadingCrops(false);
      });
  };

  const handleSvgPolygonClick = (det) => {
    setSelectedReportRowId(det.id);
    handleSelectReportCrop(det);
    
    // Scroll the table row into view
    const rowElement = document.getElementById(`report-row-${det.id}`);
    if (rowElement) {
      rowElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleSelectPdf = (filename) => {
    setSelectedPdf(filename);
    setIsPdfLoading(true);
    setPagesList([]);
    setYoloStatus('idle');
    setYoloBoxes([]);
    setCropsStatus('idle');
    setDetections([]);
    setError(null);
    setSelectedCrop(null);

    fetch(`http://localhost:8000/pdf/load?filename=${filename}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
        } else {
          setPagesList(data.pages);
          setActivePageNum(1);
        }
        setIsPdfLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setIsPdfLoading(false);
      });
  };

  const handleRunYolo = () => {
    if (!selectedPdf) return;
    setYoloStatus('running');
    setYoloBoxes([]);
    setCropsStatus('idle');
    setDetections([]);
    setError(null);

    fetch(`http://localhost:8000/pdf/page/yolo?filename=${selectedPdf}&page_num=${activePageNum}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
          setYoloStatus('idle');
        } else {
          setYoloBoxes(data.boxes);
          setYoloStatus('done');
        }
      })
      .catch(err => {
        setError(err.message);
        setYoloStatus('idle');
      });
  };

  const handleRunProcessCrops = () => {
    if (!selectedPdf) return;
    setCropsStatus('running');
    setDetections([]);
    setError(null);

    fetch(`http://localhost:8000/pdf/page/process_crops?filename=${selectedPdf}&page_num=${activePageNum}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
          setCropsStatus('idle');
        } else {
          setDetections(data.detections);
          setCropsStatus('done');
        }
      })
      .catch(err => {
        setError(err.message);
        setCropsStatus('idle');
      });
  };

  const startValidation = () => {
    if (!selectedValImage) return;
    setIsProcessing(true);
    setGtBoxes([]); setValStats(null); setError(null);
    setValStep('connecting');
    setValImageState({ image: null, w: 0, h: 0, boxes: [] });
    
    ws.current = new WebSocket('ws://localhost:8000/ws/validate');
    ws.current.onopen = () => ws.current.send(JSON.stringify({ filename: selectedValImage }));
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) { setError(data.error); setIsProcessing(false); return; }
      setValStep(data.step);
      
      switch (data.step) {
        case 'image_loaded':
          setValImageState({ image: data.image, w: data.width, h: data.height, boxes: [] });
          break;
        case 'gt_loaded':
          setGtBoxes(data.gt_boxes);
          break;
        case 'validation_done':
          setValImageState(prev => ({ ...prev, boxes: data.pred_boxes }));
          setValStats(data.stats);
          setIsProcessing(false);
          break;
        default: break;
      }
    };
    ws.current.onerror = () => { setError("WebSocket error"); setIsProcessing(false); };
    ws.current.onclose = () => setIsProcessing(false);
  };

  const stopProcessing = () => {
    if (ws.current) ws.current.close();
    setIsProcessing(false);
    setValStep('idle');
  };

  const activePage = pagesList.find(p => p.page_num === activePageNum) || null;
  const strokeW = activePage ? Math.max(3, activePage.width / 350) : 3;

  // Filtered detections based on Search, Status, and Min Confidence
  const filteredDetections = detections.filter(det => {
    // 1. Search text filter
    if (searchText && !det.pred_char.toLowerCase().includes(searchText.toLowerCase())) {
      return false;
    }
    
    // 2. Confidence filter
    if (det.class_confidence * 100 < minConfidence) {
      return false;
    }
    
    // 3. Status filter
    const hasGt = det.best_iou > 0.4;
    if (statusFilter === 'correct') {
      if (hasGt && det.is_class_correct === true && det.is_orient_correct === true) return true;
      return false;
    } else if (statusFilter === 'incorrect_class') {
      if (hasGt && det.is_class_correct === false) return true;
      return false;
    } else if (statusFilter === 'incorrect_orient') {
      if (hasGt && det.is_orient_correct === false) return true;
      return false;
    } else if (statusFilter === 'unlabeled') {
      if (!hasGt) return true;
      return false;
    }
    
    return true;
  });

  // Mouse move handler for synchronized magnifying glass
  const handleCropMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setMagnifierCoords({ x, y });
  };

  // Stepper Stage Render Helper
  const renderCropStage = (stageNum, title, imageSrc, description, isLightBg = true) => {
    return (
      <div className="flex flex-col bg-slate-950 p-4 border border-slate-850 hover:border-slate-800 rounded-xl relative shadow-md transition-all">
        <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-slate-900">
          <span className="flex items-center justify-center w-5 h-5 rounded-full bg-slate-900 border border-slate-850 text-[10px] font-black text-slate-400">
            {stageNum}
          </span>
          <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-widest">
            {title}
          </span>
        </div>
        
        <div
          className={`h-36 rounded-lg border flex items-center justify-center p-2.5 overflow-hidden relative cursor-crosshair ${
            isLightBg ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-850'
          }`}
          onMouseMove={handleCropMouseMove}
          onMouseLeave={() => setMagnifierCoords(null)}
        >
          <img
            src={imageSrc}
            alt={title}
            className="max-h-full max-w-full object-contain select-none pointer-events-none"
          />
          {magnifierCoords && (
            <>
              {/* Magnified zoom circle */}
              <div
                className={`absolute w-20 h-20 rounded-full border-2 pointer-events-none shadow-2xl overflow-hidden bg-no-repeat z-30 ${
                  isLightBg ? 'border-emerald-500 bg-white' : 'border-emerald-400 bg-slate-950'
                }`}
                style={{
                  left: `${magnifierCoords.x}%`,
                  top: `${magnifierCoords.y}%`,
                  transform: 'translate(-50%, -50%)',
                  backgroundImage: `url(${imageSrc})`,
                  backgroundSize: '300%',
                  backgroundPosition: `${magnifierCoords.x}% ${magnifierCoords.y}%`
                }}
              />
              {/* Cursor crosshair indicator */}
              <div
                className="absolute w-2.5 h-2.5 rounded-full bg-red-500 border border-white pointer-events-none z-10 shadow"
                style={{
                  left: `${magnifierCoords.x}%`,
                  top: `${magnifierCoords.y}%`,
                  transform: 'translate(-50%, -50%)'
                }}
              />
            </>
          )}
        </div>
        
        <span className="text-[10px] text-slate-400 mt-3 block leading-relaxed min-h-[48px]">
          {description}
        </span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col overflow-hidden font-sans">
      
      {/* Header Navbar */}
      <header className="h-16 border-b border-slate-800 bg-slate-900/40 backdrop-blur-md px-6 flex items-center justify-between shrink-0 z-30">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-xl border border-emerald-500/20 text-emerald-400">
            <Target className="w-6 h-6 animate-pulse" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
              Blueprint OCR Dashboard
            </span>
            <span className="text-[10px] font-semibold text-slate-500 tracking-wider uppercase">
              Vite + React Interactive Client
            </span>
          </div>
        </div>

        {/* Global Mode Switcher */}
        <div className="flex items-center gap-4">
          <div className="flex bg-slate-950 border border-slate-800 p-0.5 rounded-lg">
            <button
              onClick={() => setMode('pipeline')}
              className={`px-4 py-1.5 rounded-md text-xs font-semibold tracking-wide transition-all ${
                mode === 'pipeline'
                  ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-slate-950 font-bold shadow-md shadow-emerald-500/10'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Interactive Pipeline
            </button>
            <button
              onClick={() => setMode('validate')}
              className={`px-4 py-1.5 rounded-md text-xs font-semibold tracking-wide transition-all ${
                mode === 'validate'
                  ? 'bg-blue-600 text-white font-bold shadow-md shadow-blue-500/10'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              YOLO Validation
            </button>
            <button
              onClick={() => setMode('report')}
              className={`px-4 py-1.5 rounded-md text-xs font-semibold tracking-wide transition-all ${
                mode === 'report'
                  ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-slate-950 font-bold shadow-md shadow-amber-500/10'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Batch Reports Viewer
            </button>
          </div>
          
          {selectedPdf && mode === 'pipeline' && (
            <button
              onClick={() => {
                setSelectedPdf(null);
                setPagesList([]);
              }}
              className="px-3 py-1.5 bg-slate-800 hover:bg-red-500/20 text-slate-300 hover:text-red-400 border border-slate-700 hover:border-red-500/30 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Close Document
            </button>
          )}
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden min-h-0 relative">
        
        {/* Error Banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20, x: '-50%' }}
              animate={{ opacity: 1, y: 0, x: '-50%' }}
              exit={{ opacity: 0, y: -20, x: '-50%' }}
              className="absolute top-4 left-1/2 bg-red-500/15 border border-red-500/30 text-red-200 px-6 py-3 rounded-xl backdrop-blur-md z-50 flex items-center gap-3 shadow-2xl"
            >
              <AlertTriangle className="w-5 h-5 text-red-400" />
              <span className="text-sm font-semibold">{error}</span>
              <button onClick={() => setError(null)} className="ml-4 text-xs font-bold text-red-400 hover:text-red-300 underline">Dismiss</button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── CASE 1: PIPELINE MODE ── */}
        {mode === 'pipeline' && (
          <div className="flex-1 flex overflow-hidden">
            
            {/* If no PDF is selected, show beautiful loading splash selection */}
            {!selectedPdf ? (
              <div className="flex-grow flex items-center justify-center p-8 bg-slate-950/20 select-none overflow-y-auto">
                <div className="max-w-xl w-full flex flex-col items-center">
                  <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="w-20 h-20 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 shadow-2xl mb-6"
                  >
                    <FileText className="w-10 h-10" />
                  </motion.div>
                  
                  <h2 className="text-2xl font-extrabold text-white text-center tracking-tight mb-2">
                    Load Document to Begin
                  </h2>
                  <p className="text-sm text-slate-400 text-center max-w-sm mb-8">
                    Select one of the training or validation engineering blueprint PDFs below to start the interactive symbol recognition pipeline.
                  </p>

                  <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {pdfs.length === 0 ? (
                      <div className="col-span-full py-8 text-center text-slate-500 border border-dashed border-slate-800 rounded-2xl">
                        <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2 text-slate-655" />
                        Scanning pdfs/ folder...
                      </div>
                    ) : (
                      pdfs.map((pdfName) => (
                        <button
                          key={pdfName}
                          onClick={() => handleSelectPdf(pdfName)}
                          className="flex items-center gap-4 p-4 bg-slate-900 border border-slate-800 rounded-xl hover:border-emerald-500/40 text-left hover:bg-slate-900/80 transition-all shadow-md group"
                        >
                          <div className="p-2.5 bg-slate-950 rounded-lg text-slate-400 group-hover:text-emerald-400 group-hover:bg-emerald-500/5 border border-slate-850 transition-colors">
                            <FileText className="w-5 h-5" />
                          </div>
                          <div className="flex flex-col min-w-0">
                            <span className="font-semibold text-slate-200 text-sm truncate">{pdfName}</span>
                            <span className="text-[10px] text-slate-500 mt-0.5">Click to render pages</span>
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </div>
            ) : isPdfLoading ? (
              // Loading screen while converting PDF
              <div className="flex-grow flex flex-col items-center justify-center gap-4">
                <Loader2 className="w-12 h-12 animate-spin text-emerald-500" />
                <div className="text-center">
                  <h3 className="font-bold text-slate-200 text-lg">Converting PDF Blueprint</h3>
                  <p className="text-sm text-slate-400 mt-1">Rendering page vectors to clean pixels via PyMuPDF...</p>
                </div>
              </div>
            ) : (
              // Interactive Workspace UI once PDF pages are loaded
              <div className="flex-1 flex overflow-hidden min-h-0">
                
                {/* 1. Left Sidebar: Interactive controls */}
                <div className="w-[300px] border-r border-slate-800 bg-slate-900/20 backdrop-blur-md p-5 flex flex-col gap-6 shrink-0 overflow-y-auto custom-scrollbar z-20">
                  
                  {/* Active Document info */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest block">Active Document</span>
                    <span className="font-bold text-sm text-slate-200 block truncate mt-0.5">{selectedPdf}</span>
                    <span className="text-[10px] text-slate-400 font-semibold block mt-1">Pages Count: {pagesList.length}</span>
                  </div>

                  {/* Page selector */}
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Select Page</span>
                    <div className="grid grid-cols-4 gap-2">
                      {pagesList.map(p => (
                        <button
                          key={p.page_num}
                          onClick={() => setActivePageNum(p.page_num)}
                          className={`h-9 rounded-lg border text-xs font-bold transition-all ${
                            activePageNum === p.page_num
                              ? 'bg-emerald-500 text-slate-950 border-emerald-400 font-extrabold shadow-md'
                              : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                          }`}
                        >
                          {p.page_num}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Interactive Steps List */}
                  <div className="flex flex-col gap-4">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">Pipeline Execution</span>

                    {/* Step 1: YOLO Bounding Boxes */}
                    <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 relative">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="flex items-center justify-center w-5 h-5 rounded-full bg-slate-950 border border-slate-850 text-[10px] font-extrabold text-slate-400">1</span>
                          <span className="text-xs font-extrabold text-slate-200 tracking-tight">YOLO-OBB Scan</span>
                        </div>
                        {yoloStatus === 'done' && (
                          <span className="text-[9px] font-extrabold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">Active</span>
                        )}
                      </div>
                      
                      <p className="text-[11px] text-slate-400 leading-normal">
                        Identify symbol locations on the page using the oriented bounding box detection model.
                      </p>

                      {yoloStatus === 'running' ? (
                        <div className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg border border-slate-700 bg-slate-850 text-slate-300 font-semibold text-xs animate-pulse">
                          <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
                          Running YOLOv8...
                        </div>
                      ) : (
                        <button
                          onClick={handleRunYolo}
                          className="w-full py-2.5 rounded-lg font-bold text-xs bg-slate-950 hover:bg-slate-850 border border-slate-800 hover:border-slate-700 text-emerald-400 transition-all shadow-sm flex items-center justify-center gap-1.5"
                        >
                          <Play className="w-3.5 h-3.5" fill="currentColor" />
                          Detect symbols on Page
                        </button>
                      )}

                      {yoloStatus === 'done' && (
                        <div className="text-[11px] font-bold text-emerald-400 bg-emerald-500/5 p-2 rounded-lg border border-emerald-500/10 text-center">
                          ✓ Detected {yoloBoxes.length} potential symbol boxes!
                        </div>
                      )}
                    </div>

                    {/* Step 2: Noise Removal & Recognition */}
                    <div className={`bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 relative ${
                      yoloStatus !== 'done' && 'opacity-50 select-none pointer-events-none'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="flex items-center justify-center w-5 h-5 rounded-full bg-slate-950 border border-slate-850 text-[10px] font-extrabold text-slate-400">2</span>
                          <span className="text-xs font-extrabold text-slate-200 tracking-tight">Clean & Classify</span>
                        </div>
                        {cropsStatus === 'done' && (
                          <span className="text-[9px] font-extrabold text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded border border-teal-500/20">Success</span>
                        )}
                      </div>

                      <p className="text-[11px] text-slate-400 leading-normal">
                        Apply morphological line subtraction and noise filtering on crops, followed by classification.
                      </p>

                      {cropsStatus === 'running' ? (
                        <div className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg border border-slate-700 bg-slate-850 text-slate-300 font-semibold text-xs animate-pulse">
                          <Loader2 className="w-4 h-4 animate-spin text-teal-400" />
                          Processing crops...
                        </div>
                      ) : (
                        <button
                          onClick={handleRunProcessCrops}
                          disabled={yoloStatus !== 'done'}
                          className="w-full py-2.5 rounded-lg font-bold text-xs bg-gradient-to-r from-emerald-500 to-teal-500 text-slate-950 hover:opacity-90 disabled:opacity-50 transition-all font-extrabold shadow-md"
                        >
                          Next Step: Process Crops
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* 2. Main Center-Right Workspace */}
                <div className="flex-grow flex flex-row min-h-0 overflow-hidden">
                  
                  {/* Left Column: Page Viewer */}
                  <div className="w-[45%] min-w-[320px] bg-slate-950 flex flex-col border-r border-slate-800 relative h-full">
                    <div className="h-10 border-b border-slate-900 bg-slate-900/20 px-4 flex items-center justify-between text-xs text-slate-400 select-none shrink-0 font-medium">
                      <span>Interactive Canvas — Drag or scroll to zoom.</span>
                      <span className="bg-slate-900 px-2 py-0.5 rounded border border-slate-850 font-bold text-slate-200 text-[10px]">
                        Page {activePageNum} / {pagesList.length}
                      </span>
                    </div>

                    <div
                      ref={containerRef}
                      className="flex-grow bg-slate-950 overflow-hidden relative cursor-crosshair flex items-center justify-center"
                      onMouseMove={(e) => {
                        if (!containerRef.current) return;
                        const rect = containerRef.current.getBoundingClientRect();
                        const x = ((e.clientX - rect.left) / rect.width) * 100;
                        const y = ((e.clientY - rect.top) / rect.height) * 100;
                        setMousePos({ x, y });
                      }}
                      onWheel={(e) => {
                        const zoomDelta = e.deltaY < 0 ? 0.25 : -0.25;
                        setZoom(z => Math.min(Math.max(1, z + zoomDelta), 8));
                      }}
                      onDoubleClick={() => setZoom(1)}
                    >
                      <div
                        className="absolute inset-0 w-full h-full transition-transform duration-100 ease-out flex items-center justify-center p-4"
                        style={{
                          transformOrigin: `${mousePos.x}% ${mousePos.y}%`,
                          transform: `scale(${zoom})`
                        }}
                      >
                        {activePage && (
                          <div 
                            className="relative shadow-2xl"
                            style={{ 
                              aspectRatio: `${activePage.width} / ${activePage.height}`,
                              width: 'auto',
                              height: 'auto',
                              maxWidth: '100%',
                              maxHeight: '100%'
                            }}
                          >
                            <img
                              src={activePage.image}
                              alt={`Page ${activePageNum}`}
                              className="w-full h-full block opacity-95 select-none pointer-events-none border border-slate-800"
                            />
                            
                            <svg
                              viewBox={`0 0 ${activePage.width} ${activePage.height}`}
                              className="absolute inset-0 w-full h-full pointer-events-none z-10"
                            >
                              {/* Overlay YOLO boxes (cyan) if done */}
                              {yoloStatus === 'done' && cropsStatus !== 'done' && yoloBoxes.map((box) => (
                                <polygon
                                  key={`yolo-${box.id}`}
                                  points={box.corners.map(p => `${p[0]},${p[1]}`).join(' ')}
                                  fill="rgba(6, 182, 212, 0.08)"
                                  stroke="#06b6d4"
                                  strokeWidth={2}
                                  vectorEffect="non-scaling-stroke"
                                  className="pointer-events-auto cursor-pointer hover:fill-cyan-500/25 transition-all"
                                  onMouseEnter={() => setHoveredBox({ type: 'YOLO Crop', corners: box.corners, conf: box.conf })}
                                  onMouseLeave={() => setHoveredBox(null)}
                                />
                              ))}

                              {/* Overlay Classified crops (green) if done */}
                              {cropsStatus === 'done' && detections.map((det) => {
                                const isSelected = selectedCrop?.id === det.id;
                                return (
                                  <polygon
                                    key={`det-${det.id}`}
                                    points={det.corners.map(p => `${p[0]},${p[1]}`).join(' ')}
                                    fill={isSelected ? "rgba(16, 185, 129, 0.25)" : "rgba(16, 185, 129, 0.08)"}
                                    stroke={isSelected ? "#10b981" : "#10b981"}
                                    strokeWidth={isSelected ? 3 : 2}
                                    vectorEffect="non-scaling-stroke"
                                    className="pointer-events-auto cursor-pointer hover:fill-emerald-500/25 transition-all"
                                    onClick={() => setSelectedCrop(det)}
                                    onMouseEnter={() => setHoveredBox({
                                      type: 'Symbol',
                                      corners: det.corners,
                                      conf: det.class_confidence,
                                      char: det.pred_char
                                    })}
                                    onMouseLeave={() => setHoveredBox(null)}
                                  />
                                );
                              })}
                            </svg>
                          </div>
                        )}
                      </div>

                      {/* Tooltip Overlay */}
                      <AnimatePresence>
                        {hoveredBox && (
                          <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            style={{
                              left: `${mousePos.x}%`,
                              top: `${mousePos.y}%`,
                              transform: 'translate(-50%, -120%)'
                            }}
                            className="absolute bg-slate-950/95 border border-slate-800 backdrop-blur-md px-3 py-2 rounded-xl shadow-2xl z-40 pointer-events-none flex flex-col gap-0.5 text-xs text-white"
                          >
                            <span className="font-bold text-slate-400 text-[10px] uppercase">{hoveredBox.type}</span>
                            {hoveredBox.char && (
                              <span className="text-amber-400 font-mono text-sm font-black">Symbol: '{hoveredBox.char}'</span>
                            )}
                            <span className="font-semibold text-slate-200">
                              Conf: {(hoveredBox.conf * 100).toFixed(1)}%
                            </span>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>

                  {/* Right Column: Compact Grid containing comparison crops */}
                  <div className="flex-1 bg-slate-950 p-6 overflow-y-scroll custom-scrollbar flex flex-col min-h-0 select-none h-full border-l border-slate-900">
                    
                    {/* If crops are not loaded/processed, show helpful instructions */}
                    {cropsStatus !== 'done' ? (
                      <div className="flex-grow flex flex-col items-center justify-center text-slate-600 text-center py-12 border border-dashed border-slate-900 rounded-2xl">
                        <LayoutGrid className="w-10 h-10 mb-3 text-slate-800" />
                        <h4 className="font-bold text-slate-500 text-sm">Crops Recognition Grid</h4>
                        <p className="text-xs max-w-sm mt-1">
                          After detecting boxes in Step 1, click "Next Step: Process Crops" to view adaptive-thresholded noise removal and classification side-by-side in this panel.
                        </p>
                      </div>
                    ) : (
                      // Grid layout showing Raw vs Cleaned and recognition labels
                      <div className="flex flex-col gap-4">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                            Processed Bounding Box Crops Grid (Before vs After)
                          </span>
                          <span className="bg-slate-900 text-emerald-400 text-[10px] px-2 py-0.5 border border-slate-850 rounded-full font-bold">
                            {filteredDetections.length} of {detections.length} matches shown
                          </span>
                        </div>

                        {/* Filters Toolbar */}
                        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
                          <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
                            {/* Search */}
                            <div className="relative w-full sm:w-44">
                              <input
                                type="text"
                                placeholder="Search symbol character..."
                                value={searchText}
                                onChange={(e) => setSearchText(e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 hover:border-slate-700 focus:border-emerald-500 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none transition-all"
                              />
                            </div>
                            
                            {/* Status filter */}
                            <select
                              value={statusFilter}
                              onChange={(e) => setStatusFilter(e.target.value)}
                              className="bg-slate-950 border border-slate-800 hover:border-slate-700 focus:border-emerald-500 text-xs text-slate-350 px-2.5 py-1.5 rounded-lg focus:outline-none cursor-pointer transition-all font-semibold"
                            >
                              <option value="all">All Statuses</option>
                              <option value="correct">✓ Fully Correct</option>
                              <option value="incorrect_class">✗ Misclassified</option>
                              <option value="incorrect_orient">✗ Incorrect Orientation</option>
                              <option value="unlabeled">No Ground Truth</option>
                            </select>
                          </div>
                          
                          {/* Confidence Slider */}
                          <div className="flex items-center gap-3 w-full md:w-auto shrink-0">
                            <span className="text-[11px] font-bold text-slate-400 min-w-[95px]">Min Conf: {minConfidence}%</span>
                            <input
                              type="range"
                              min="0"
                              max="100"
                              value={minConfidence}
                              onChange={(e) => setMinConfidence(Number(e.target.value))}
                              className="w-full sm:w-28 accent-emerald-500 h-1 bg-slate-800 rounded-lg cursor-pointer"
                            />
                            {minConfidence > 0 && (
                              <button
                                onClick={() => setMinConfidence(0)}
                                className="text-[10px] text-emerald-450 hover:text-emerald-400 font-bold underline"
                              >
                                Reset
                              </button>
                            )}
                          </div>
                        </div>

                        {filteredDetections.length === 0 ? (
                          <div className="py-12 text-center text-slate-500 border border-dashed border-slate-900 rounded-2xl">
                            No matching crops found. Try adjusting filters.
                          </div>
                        ) : (
                          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                            {filteredDetections.map((det) => {
                              const isHighConf = det.class_confidence >= 0.8;
                              const isSelected = selectedCrop?.id === det.id;
                              
                              // Correctness colors
                              const hasGt = det.best_iou > 0.4;
                              const isCorrect = hasGt && det.is_class_correct === true && det.is_orient_correct === true;
                              const isClassWrong = hasGt && det.is_class_correct === false;
                              const isOrientWrong = hasGt && det.is_orient_correct === false;

                              let cardBorderColor = 'border-slate-800 hover:border-slate-700 bg-slate-900/60';
                              if (isSelected) {
                                cardBorderColor = 'border-emerald-500 bg-emerald-950/10 ring-2 ring-emerald-500/10';
                              } else if (hasGt) {
                                if (isCorrect) {
                                  cardBorderColor = 'border-emerald-500/20 hover:border-emerald-550/40 bg-emerald-950/5';
                                } else if (isClassWrong) {
                                  cardBorderColor = 'border-red-500/20 hover:border-red-550/40 bg-red-950/5';
                                } else if (isOrientWrong) {
                                  cardBorderColor = 'border-amber-500/20 hover:border-amber-550/40 bg-amber-950/5';
                                }
                              }

                              return (
                                <motion.div
                                  key={det.id}
                                  whileHover={{ y: -3, scale: 1.01 }}
                                  onClick={() => setSelectedCrop(det)}
                                  className={`border rounded-xl p-3 flex flex-col gap-3 cursor-pointer shadow-md transition-all ${cardBorderColor}`}
                                >
                                  {/* Card Header */}
                                  <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 border-b border-slate-850 pb-1.5 shrink-0">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-slate-300">Box #{det.id + 1}</span>
                                      {hasGt ? (
                                        isCorrect ? (
                                          <span className="text-[9px] font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.2 rounded border border-emerald-500/15">
                                            ✓ Match
                                          </span>
                                        ) : isClassWrong ? (
                                          <span className="text-[9px] font-bold text-red-400 bg-red-500/10 px-1.5 py-0.2 rounded border border-red-500/15">
                                            ✗ Class
                                          </span>
                                        ) : (
                                          <span className="text-[9px] font-bold text-amber-400 bg-amber-500/10 px-1.5 py-0.2 rounded border border-amber-500/15">
                                            ↻ Rotation
                                          </span>
                                        )
                                      ) : (
                                        <span className="text-[9px] font-bold text-slate-500 bg-slate-950 px-1.5 py-0.2 rounded border border-slate-800">
                                          Unlabeled
                                        </span>
                                      )}
                                    </div>
                                    <span className="text-slate-500">YOLO: {(det.yolo_conf * 100).toFixed(0)}%</span>
                                  </div>

                                  {/* Comparison crops */}
                                  <div className="grid grid-cols-2 gap-2 h-20 bg-slate-950 p-1.5 rounded-lg border border-slate-850 shrink-0">
                                    {/* Left: Raw crop */}
                                    <div className="relative flex flex-col items-center justify-center bg-slate-900 rounded overflow-hidden p-0.5 border border-slate-800">
                                      <img src={det.crops.raw} alt="Raw crop" className="max-h-10 max-w-[85%] object-contain" />
                                      <span className="absolute bottom-0.5 left-1 text-[7px] font-bold bg-slate-950/80 px-1 text-slate-500 rounded">Raw</span>
                                    </div>

                                    {/* Right: Clean Rectified crop */}
                                    <div className="relative flex flex-col items-center justify-center bg-white rounded overflow-hidden p-0.5 border border-slate-200">
                                      <img src={det.crops.rect} alt="Rectified crop" className="max-h-10 max-w-[85%] object-contain" />
                                      <span className="absolute bottom-0.5 right-1 text-[7px] font-bold bg-white/95 border px-1 text-slate-655 rounded shadow-sm">Rectified</span>
                                    </div>
                                  </div>

                                  {/* Prediction tag */}
                                  <div className="flex items-center justify-between mt-0.5 leading-none">
                                    <div className="flex flex-col gap-0.5">
                                      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Prediction</span>
                                      <span className="font-mono text-base font-extrabold text-amber-400">'{det.pred_char}'</span>
                                    </div>
                                    <div className="flex flex-col items-end gap-0.5">
                                      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Confidence</span>
                                      <span className={`text-[11px] font-bold px-1.5 py-0.5 rounded ${
                                        isHighConf ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                                      }`}>
                                        {(det.class_confidence * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  </div>
                                </motion.div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── CASE 2: VALIDATION MODE ── */}
        {mode === 'validate' && (
          <div className="flex-grow flex flex-col lg:flex-row gap-6 p-6 overflow-y-auto custom-scrollbar">
            
            {/* Sidebar list of validation images */}
            <div className="w-full lg:w-64 bg-slate-900 border border-slate-800 p-4 rounded-2xl flex flex-col gap-4 shrink-0">
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-2">
                Validation Images
              </h3>
              
              <div className="flex flex-col gap-2 max-h-[220px] lg:max-h-none overflow-y-auto pr-1 custom-scrollbar">
                {valImages.length === 0 ? (
                  <span className="text-xs text-slate-500">Scanning dataset_yolo...</span>
                ) : (
                  valImages.map(imgName => (
                    <button
                      key={imgName}
                      onClick={() => !isProcessing && setSelectedValImage(imgName)}
                      className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg border text-left text-xs font-semibold truncate transition-all ${
                        selectedValImage === imgName
                          ? 'bg-blue-600/15 border-blue-500 text-blue-300'
                          : 'bg-slate-950 border-slate-850 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <ImageIcon className="w-4 h-4 shrink-0 text-slate-550" />
                      <span className="truncate">{imgName}</span>
                    </button>
                  ))
                )}
              </div>

              {isProcessing ? (
                <button
                  onClick={stopProcessing}
                  className="w-full py-2.5 bg-red-650 hover:bg-red-500 text-white font-bold text-xs rounded-xl shadow-lg transition-all"
                >
                  Stop validation
                </button>
              ) : (
                <button
                  onClick={startValidation}
                  disabled={!selectedValImage}
                  className={`w-full py-2.5 rounded-xl font-bold text-xs shadow-lg transition-all ${
                    selectedValImage
                      ? 'bg-blue-600 hover:bg-blue-500 text-white font-bold shadow-blue-500/20'
                      : 'bg-slate-850 text-slate-500 cursor-not-allowed border border-slate-850'
                  }`}
                >
                  Run Validation
                </button>
              )}
            </div>

            {/* Validation canvas viewer */}
            {valImageState.image ? (
              <div className="flex-1 flex flex-col lg:flex-row gap-6 min-h-0">
                <div className="flex-grow bg-slate-900 border border-slate-850 rounded-2xl overflow-hidden flex flex-col relative">
                  <div className="h-10 border-b border-slate-850 bg-slate-950/30 px-4 flex items-center justify-between text-xs text-slate-400 select-none">
                    <span>Validation Image view — Blue = Ground Truth | Dashed Green = Prediction</span>
                    <span className="bg-slate-850 text-slate-300 px-2 py-0.5 rounded border border-slate-800 text-[10px] font-semibold">
                      Dim: {valImageState.w}x{valImageState.h}
                    </span>
                  </div>

                  <div
                    ref={containerRef}
                    className="flex-grow bg-slate-950 relative overflow-hidden min-h-[350px] flex items-center justify-center"
                    onMouseMove={(e) => {
                      if (!containerRef.current) return;
                      const rect = containerRef.current.getBoundingClientRect();
                      const x = ((e.clientX - rect.left) / rect.width) * 100;
                      const y = ((e.clientY - rect.top) / rect.height) * 100;
                      setMousePos({ x, y });
                    }}
                    onWheel={(e) => {
                      const zoomDelta = e.deltaY < 0 ? 0.25 : -0.25;
                      setZoom(z => Math.min(Math.max(1, z + zoomDelta), 8));
                    }}
                    onDoubleClick={() => setZoom(1)}
                  >
                    <div
                      className="absolute inset-0 w-full h-full transition-transform duration-100 ease-out flex items-center justify-center p-4"
                      style={{
                        transformOrigin: `${mousePos.x}% ${mousePos.y}%`,
                        transform: `scale(${zoom})`
                      }}
                    >
                      <div 
                        className="relative shadow-2xl"
                        style={{
                          aspectRatio: `${valImageState.w} / ${valImageState.h}`,
                          width: 'auto',
                          height: 'auto',
                          maxWidth: '100%',
                          maxHeight: '100%'
                        }}
                      >
                        <img 
                          src={valImageState.image} 
                          alt="Validation view" 
                          className="w-full h-full block opacity-95 select-none pointer-events-none" 
                        />
                        <svg
                          viewBox={`0 0 ${valImageState.w} ${valImageState.h}`}
                          className="absolute inset-0 w-full h-full pointer-events-none z-10"
                        >
                          {gtBoxes.map((box, i) => (
                            <polygon 
                              key={`gt-${i}`}
                              points={box.map(p => `${p[0]},${p[1]}`).join(' ')} 
                              fill="rgba(59, 130, 246, 0.12)"
                              stroke="#3b82f6"
                              strokeWidth={2}
                              vectorEffect="non-scaling-stroke"
                              className="pointer-events-auto cursor-pointer hover:fill-blue-500/40 transition-colors"
                              onMouseEnter={() => setHoveredBox({ type: 'Ground Truth', corners: box, conf: 1.0 })}
                              onMouseLeave={() => setHoveredBox(null)}
                            />
                          ))}
                          {valImageState.boxes.map((box, i) => (
                            <polygon
                              key={`box-${box.id || i}`}
                              points={box.corners.map(p => `${p[0]},${p[1]}`).join(' ')} 
                              fill="transparent"
                              stroke="#10b981"
                              strokeWidth={2}
                              vectorEffect="non-scaling-stroke"
                              strokeDasharray="6 4"
                              className="pointer-events-auto cursor-pointer hover:fill-emerald-500/35 transition-colors animate-pulse"
                              onMouseEnter={() => setHoveredBox({ type: 'YOLO Prediction', corners: box.corners, conf: box.conf })}
                              onMouseLeave={() => setHoveredBox(null)}
                            />
                          ))}
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Validation Metrics Panel */}
                <div className="w-full lg:w-72 bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col gap-5 shrink-0 select-none">
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800 pb-2 flex items-center gap-1.5">
                    <Crosshair className="w-4 h-4 text-blue-400" /> Metrics Report
                  </h3>

                  {!valStats ? (
                    <div className="flex-1 flex flex-col items-center justify-center text-slate-600 gap-3 py-16">
                      <Loader2 className="w-6 h-6 animate-spin text-blue-550" />
                      <span className="text-[11px] font-bold">Calculating stats...</span>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-4">
                      <div className="bg-slate-950 p-4 rounded-xl border border-blue-500/15">
                        <span className="text-[10px] text-slate-500 block">Average Intersection Over Union (IoU)</span>
                        <span className="text-3xl font-black text-blue-400 block mt-1">{(valStats.avg_iou * 100).toFixed(1)}%</span>
                      </div>

                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-850 flex flex-col gap-2">
                        <div className="flex justify-between text-xs font-semibold">
                          <span className="text-slate-500">Actual Targets</span>
                          <span className="text-slate-200">{valStats.total_gt}</span>
                        </div>
                        <div className="flex justify-between text-xs font-semibold">
                          <span className="text-slate-500">YOLO Predictions</span>
                          <span className="text-slate-200">{valStats.total_pred}</span>
                        </div>
                      </div>

                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-850 flex flex-col gap-3">
                        <div className="flex justify-between text-xs font-bold items-center leading-none">
                          <span className="text-slate-500">True Positives</span>
                          <span className="text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                            {valStats.true_positives}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs font-bold items-center leading-none">
                          <span className="text-slate-500 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3 text-amber-500" /> False Positives
                          </span>
                          <span className="text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">
                            {valStats.false_positives}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs font-bold items-center leading-none">
                          <span className="text-slate-500 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3 text-red-500" /> Missed Symbols
                          </span>
                          <span className="text-red-400 bg-red-500/10 px-1.5 py-0.5 rounded">
                            {valStats.false_negatives}
                          </span>
                        </div>
                      </div>

                      <div className="flex flex-col gap-2 text-[10px] font-semibold text-slate-400 p-3 bg-slate-950/20 border border-slate-850 rounded-lg">
                        <div className="flex items-center gap-2">
                          <div className="w-3.5 h-3.5 rounded bg-blue-500/20 border border-blue-400" />
                          <span>Ground Truth (Solid Blue)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3.5 h-3.5 rounded border border-dashed border-emerald-400" />
                          <span>Predictions (Dashed Green)</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex-grow flex flex-col items-center justify-center text-slate-600 gap-3 py-16">
                <ImageIcon className="w-12 h-12 text-slate-800" />
                <h4 className="font-bold text-slate-500 text-sm">No Validation Image Selected</h4>
                <p className="text-xs max-w-sm mt-0.5 text-center">
                  Select a validation sample from the left panel and click "Run Validation" to assess oriented bounding box recall metrics.
                </p>
              </div>
            )}
          </div>
        )}

        {/* ── CASE 3: BATCH REPORT VIEWERS MODE ── */}
        {mode === 'report' && (
          <div className="flex-grow flex overflow-hidden min-h-0 relative">
            
            {/* Sidebar list of reports */}
            <div className="w-[240px] border-r border-slate-800 bg-slate-900/20 backdrop-blur-md p-4 flex flex-col gap-4 shrink-0 overflow-y-auto custom-scrollbar z-20">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">Available Reports</span>
                <button 
                  onClick={() => {
                    fetch('http://localhost:8000/reports')
                      .then(res => res.json())
                      .then(data => { if(data.reports) setReportsList(data.reports); })
                      .catch(err => console.error(err));
                  }}
                  className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors"
                  title="Refresh Reports List"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="flex flex-col gap-2">
                {reportsList.length === 0 ? (
                  <span className="text-xs text-slate-500 italic py-4 text-center">No reports found</span>
                ) : (
                  reportsList.map(rName => (
                    <button
                      key={rName}
                      onClick={() => handleSelectReport(rName)}
                      className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg border text-left text-xs font-semibold truncate transition-all ${
                        selectedReportName === rName
                          ? 'bg-amber-600/15 border-amber-500 text-amber-300'
                          : 'bg-slate-950 border-slate-850 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Activity className="w-4 h-4 shrink-0 text-slate-550" />
                      <span className="truncate">{rName}</span>
                    </button>
                  ))
                )}
              </div>
            </div>

            {/* Main content container */}
            <div className="flex-grow flex flex-col overflow-hidden min-h-0">
              {!selectedReportName ? (
                <div className="flex-grow flex items-center justify-center p-8 bg-slate-950/20 select-none overflow-y-auto">
                  <div className="max-w-xl w-full flex flex-col items-center">
                    <motion.div
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="w-20 h-20 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400 shadow-2xl mb-6"
                    >
                      <FileText className="w-10 h-10" />
                    </motion.div>
                    
                    <h2 className="text-2xl font-extrabold text-white text-center tracking-tight mb-2">
                      Select Batch Report
                    </h2>
                    <p className="text-sm text-slate-400 text-center max-w-sm mb-8">
                      Choose an offline-generated evaluation report from the left panel to explore detailed metric analytics and interactive visual overlays.
                    </p>
                  </div>
                </div>
              ) : isReportLoading ? (
                <div className="flex-grow flex flex-col items-center justify-center gap-4">
                  <Loader2 className="w-12 h-12 animate-spin text-amber-500" />
                  <div className="text-center">
                    <h3 className="font-bold text-slate-200 text-lg">Loading Report Data</h3>
                    <p className="text-sm text-slate-400 mt-1">Fetching metrics, predictions, and annotated visual output...</p>
                  </div>
                </div>
              ) : activeReportData && (() => {
                // Filtered report detections
                const filteredReportDetections = activeReportData.details.filter(det => {
                  if (reportSearchText && !det.prediction.toLowerCase().includes(reportSearchText.toLowerCase())) {
                    return false;
                  }
                  
                  if (det.class_conf * 100 < reportMinConfidence) {
                    return false;
                  }
                  
                  const hasGt = det.iou > 0.4;
                  if (reportStatusFilter === 'correct') {
                    if (hasGt && det.is_class_correct === true && det.is_orient_correct === true) return true;
                    return false;
                  } else if (reportStatusFilter === 'incorrect_class') {
                    if (hasGt && det.is_class_correct === false) return true;
                    return false;
                  } else if (reportStatusFilter === 'incorrect_orient') {
                    if (hasGt && det.is_orient_correct === false) return true;
                    return false;
                  } else if (reportStatusFilter === 'unlabeled') {
                    if (!hasGt) return true;
                    return false;
                  }
                  
                  return true;
                });

                // Sorted report detections
                const sortedReportDetections = [...filteredReportDetections].sort((a, b) => {
                  let aVal = a[reportSortField];
                  let bVal = b[reportSortField];
                  
                  if (reportSortField === 'id') {
                    aVal = a.id;
                    bVal = b.id;
                  } else if (reportSortField === 'yolo_conf') {
                    aVal = a.yolo_conf;
                    bVal = b.yolo_conf;
                  } else if (reportSortField === 'class_conf') {
                    aVal = a.class_conf;
                    bVal = b.class_conf;
                  } else if (reportSortField === 'pred_angle') {
                    aVal = a.pred_angle;
                    bVal = b.pred_angle;
                  } else if (reportSortField === 'iou') {
                    aVal = a.iou;
                    bVal = b.iou;
                  }
                  
                  if (aVal === undefined || aVal === null) return reportSortAsc ? 1 : -1;
                  if (bVal === undefined || bVal === null) return reportSortAsc ? -1 : 1;
                  
                  if (aVal < bVal) return reportSortAsc ? -1 : 1;
                  if (aVal > bVal) return reportSortAsc ? 1 : -1;
                  return 0;
                });

                const toggleReportSort = (field) => {
                  if (reportSortField === field) {
                    setReportSortAsc(!reportSortAsc);
                  } else {
                    setReportSortField(field);
                    setReportSortAsc(true);
                  }
                };

                const renderSortIcon = (field) => {
                  if (reportSortField !== field) return <ArrowUpDown className="w-3 h-3 text-slate-600 inline ml-1" />;
                  return reportSortAsc 
                    ? <ChevronUp className="w-3.5 h-3.5 text-amber-500 inline ml-0.5" /> 
                    : <ChevronDown className="w-3.5 h-3.5 text-amber-500 inline ml-0.5" />;
                };

                return (
                  <div className="flex-grow flex flex-col overflow-hidden min-h-0">
                    
                    {/* Top Summary Cards */}
                    <div className="p-4 bg-slate-900/40 border-b border-slate-800 grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0 select-none">
                      
                      {/* Detections card */}
                      <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-850">
                        <span className="text-[10px] text-slate-500 block font-bold uppercase tracking-wider">Detected Symbols</span>
                        <span className="text-2xl font-black text-amber-450 block mt-1">{activeReportData.detections_count}</span>
                        <span className="text-[10px] text-slate-400 block mt-0.5 font-medium truncate">File: {activeReportData.image}</span>
                      </div>

                      {/* Ground truth match card */}
                      <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-850">
                        <span className="text-[10px] text-slate-500 block font-bold uppercase tracking-wider">Ground Truth Match</span>
                        <span className="text-2xl font-black text-slate-200 block mt-1">
                          {activeReportData.ground_truth_count > 0 ? (
                            `${activeReportData.matched_detections} / ${activeReportData.ground_truth_count}`
                          ) : (
                            "0 (Unlabeled)"
                          )}
                        </span>
                        <span className="text-[10px] text-slate-400 block mt-0.5 font-medium">
                          {activeReportData.ground_truth_count > 0 
                            ? `${((activeReportData.matched_detections / activeReportData.ground_truth_count) * 100).toFixed(0)}% OBB Recall`
                            : "No GT comparison data"
                          }
                        </span>
                      </div>

                      {/* Classification Accuracy card */}
                      <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-855">
                        <span className="text-[10px] text-slate-500 block font-bold uppercase tracking-wider">Classification Accuracy</span>
                        <span className={`text-2xl font-black block mt-1 ${activeReportData.ground_truth_count > 0 ? 'text-emerald-400' : 'text-slate-500'}`}>
                          {activeReportData.ground_truth_count > 0 
                            ? `${(activeReportData.classification_accuracy * 100).toFixed(1)}%` 
                            : "N/A"
                          }
                        </span>
                        <span className="text-[10px] text-slate-400 block mt-0.5 font-medium">
                          {activeReportData.ground_truth_count > 0 
                            ? `${activeReportData.correct_classifications} / ${activeReportData.matched_detections} correct`
                            : "No labels matched"
                          }
                        </span>
                      </div>

                      {/* Orientation Accuracy card */}
                      <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-855">
                        <span className="text-[10px] text-slate-500 block font-bold uppercase tracking-wider">Orientation Score</span>
                        <span className={`text-2xl font-black block mt-1 ${activeReportData.ground_truth_count > 0 ? 'text-emerald-400' : 'text-slate-500'}`}>
                          {activeReportData.ground_truth_count > 0 
                            ? `${(activeReportData.orientation_accuracy * 100).toFixed(1)}%` 
                            : "N/A"
                          }
                        </span>
                        <span className="text-[10px] text-slate-400 block mt-0.5 font-medium">
                          {activeReportData.ground_truth_count > 0 
                            ? `${activeReportData.correct_orientations} / ${activeReportData.matched_detections} correct`
                            : "No labels matched"
                          }
                        </span>
                      </div>

                    </div>

                    {/* Split Pane: Left Page Image / Right Scrollable Table */}
                    <div className="flex-grow flex overflow-hidden min-h-0">
                      
                      {/* Left Page Image with interactive SVG overlay */}
                      <div className="w-[45%] bg-slate-950 flex flex-col border-r border-slate-800 relative h-full">
                        <div className="h-10 border-b border-slate-900 bg-slate-900/20 px-4 flex items-center justify-between text-xs text-slate-400 select-none shrink-0 font-medium">
                          <span>Annotated View — Drag/scroll to zoom. Click boxes to view steps.</span>
                          <span className="bg-slate-900 px-2 py-0.5 rounded border border-slate-850 font-bold text-slate-200 text-[10px]">
                            SVG Coordinate Matcher
                          </span>
                        </div>

                        <div
                          ref={containerRef}
                          className="flex-grow bg-slate-950 overflow-hidden relative cursor-crosshair flex items-center justify-center"
                          onMouseMove={(e) => {
                            if (!containerRef.current) return;
                            const rect = containerRef.current.getBoundingClientRect();
                            const x = ((e.clientX - rect.left) / rect.width) * 100;
                            const y = ((e.clientY - rect.top) / rect.height) * 100;
                            setMousePos({ x, y });
                          }}
                          onWheel={(e) => {
                            const zoomDelta = e.deltaY < 0 ? 0.25 : -0.25;
                            setZoom(z => Math.min(Math.max(1, z + zoomDelta), 8));
                          }}
                          onDoubleClick={() => setZoom(1)}
                        >
                          <div
                            className="absolute inset-0 w-full h-full transition-transform duration-100 ease-out flex items-center justify-center p-4"
                            style={{
                              transformOrigin: `${mousePos.x}% ${mousePos.y}%`,
                              transform: `scale(${zoom})`
                            }}
                          >
                            {activeReportImage && (
                              <div 
                                className="relative shadow-2xl"
                                style={{ 
                                  aspectRatio: `${reportImageSize.w} / ${reportImageSize.h}`,
                                  width: 'auto',
                                  height: 'auto',
                                  maxWidth: '100%',
                                  maxHeight: '100%'
                                }}
                              >
                                <img
                                  src={activeReportImage}
                                  alt="Batch Report page"
                                  className="w-full h-full block opacity-95 select-none pointer-events-none border border-slate-800"
                                  onLoad={(e) => {
                                    setReportImageSize({
                                      w: e.target.naturalWidth || 1000,
                                      h: e.target.naturalHeight || 1000
                                    });
                                  }}
                                />
                                
                                <svg
                                  viewBox={`0 0 ${reportImageSize.w} ${reportImageSize.h}`}
                                  className="absolute inset-0 w-full h-full pointer-events-none z-10"
                                >
                                  {filteredReportDetections.map((det) => {
                                    const isSelected = selectedReportRowId === det.id;
                                    const isHovered = hoveredReportBox === det.id;
                                    
                                    // Color code boundary highlighting
                                    const hasGt = det.iou > 0.4;
                                    const isCorrect = hasGt && det.is_class_correct === true && det.is_orient_correct === true;
                                    const isClassWrong = hasGt && det.is_class_correct === false;
                                    const isOrientWrong = hasGt && det.is_orient_correct === false;
                                    
                                    let strokeColor = "#06b6d4"; // Cyan default
                                    let fillColor = "rgba(6, 182, 212, 0.05)";
                                    
                                    if (hasGt) {
                                      if (isCorrect) {
                                        strokeColor = "#10b981"; // Green
                                        fillColor = isHovered || isSelected ? "rgba(16, 185, 129, 0.25)" : "rgba(16, 185, 129, 0.05)";
                                      } else if (isClassWrong) {
                                        strokeColor = "#ef4444"; // Red
                                        fillColor = isHovered || isSelected ? "rgba(239, 68, 68, 0.25)" : "rgba(239, 68, 68, 0.05)";
                                      } else if (isOrientWrong) {
                                        strokeColor = "#f59e0b"; // Orange/Amber
                                        fillColor = isHovered || isSelected ? "rgba(245, 158, 11, 0.25)" : "rgba(245, 158, 11, 0.05)";
                                      }
                                    } else {
                                      fillColor = isHovered || isSelected ? "rgba(6, 182, 212, 0.25)" : "rgba(6, 182, 212, 0.05)";
                                    }

                                    return (
                                      <polygon
                                        key={`report-poly-${det.id}`}
                                        points={det.corners.map(p => `${p[0]},${p[1]}`).join(' ')}
                                        fill={fillColor}
                                        stroke={strokeColor}
                                        strokeWidth={isSelected || isHovered ? 3.5 : 2}
                                        vectorEffect="non-scaling-stroke"
                                        className="pointer-events-auto cursor-pointer transition-all"
                                        onClick={() => handleSvgPolygonClick(det)}
                                        onMouseEnter={() => {
                                          setHoveredReportBox(det.id);
                                          setHoveredBox({
                                            type: `Crop ID #${det.id}`,
                                            corners: det.corners,
                                            conf: det.class_conf,
                                            char: det.prediction
                                          });
                                        }}
                                        onMouseLeave={() => {
                                          setHoveredReportBox(null);
                                          setHoveredBox(null);
                                        }}
                                      />
                                    );
                                  })}
                                </svg>
                              </div>
                            )}
                          </div>

                          {/* Tooltip Overlay */}
                          <AnimatePresence>
                            {hoveredBox && (
                              <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                style={{
                                  left: `${mousePos.x}%`,
                                  top: `${mousePos.y}%`,
                                  transform: 'translate(-50%, -120%)'
                                }}
                                className="absolute bg-slate-950/95 border border-slate-800 backdrop-blur-md px-3 py-2 rounded-xl shadow-2xl z-40 pointer-events-none flex flex-col gap-0.5 text-xs text-white"
                              >
                                <span className="font-bold text-slate-400 text-[10px] uppercase">{hoveredBox.type}</span>
                                {hoveredBox.char && (
                                  <span className="text-amber-400 font-mono text-sm font-black">Symbol: '{hoveredBox.char}'</span>
                                )}
                                <span className="font-semibold text-slate-200">
                                  Conf: {(hoveredBox.conf * 100).toFixed(1)}%
                                </span>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      </div>

                      {/* Right Pane: Table List of detections with column sorting & filters */}
                      <div className="flex-grow flex flex-col overflow-hidden h-full bg-slate-950 border-l border-slate-900">
                        
                        {/* Filter Toolbar */}
                        <div className="p-4 border-b border-slate-900 bg-slate-900/10 flex flex-wrap gap-4 items-center justify-between select-none">
                          <div className="flex items-center gap-3">
                            {/* Search prediction */}
                            <input
                              type="text"
                              placeholder="Filter symbol character..."
                              value={reportSearchText}
                              onChange={(e) => setReportSearchText(e.target.value)}
                              className="bg-slate-950 border border-slate-800 hover:border-slate-700 focus:border-amber-500 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none transition-all w-40"
                            />

                            {/* Status select dropdown */}
                            <select
                              value={reportStatusFilter}
                              onChange={(e) => setReportStatusFilter(e.target.value)}
                              className="bg-slate-950 border border-slate-800 hover:border-slate-700 focus:border-amber-500 text-xs text-slate-350 px-2.5 py-1.5 rounded-lg focus:outline-none cursor-pointer transition-all font-semibold"
                            >
                              <option value="all">All Evaluated Statuses</option>
                              <option value="correct">✓ Fully Correct</option>
                              <option value="incorrect_class">✗ Misclassified</option>
                              <option value="incorrect_orient">✗ Incorrect Rotation</option>
                              <option value="unlabeled">No Ground Truth</option>
                            </select>
                          </div>

                          {/* Confidence range slider */}
                          <div className="flex items-center gap-2">
                            <span className="text-[11px] font-bold text-slate-400 min-w-[95px]">Min Class Conf: {reportMinConfidence}%</span>
                            <input
                              type="range"
                              min="0"
                              max="100"
                              value={reportMinConfidence}
                              onChange={(e) => setReportMinConfidence(Number(e.target.value))}
                              className="w-24 accent-amber-500 h-1 bg-slate-800 rounded-lg cursor-pointer"
                            />
                            {reportMinConfidence > 0 && (
                              <button
                                onClick={() => setReportMinConfidence(0)}
                                className="text-[10px] text-amber-450 hover:text-amber-400 font-bold underline"
                              >
                                Reset
                              </button>
                            )}
                          </div>
                        </div>

                        {/* Scrollable table container */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar">
                          {sortedReportDetections.length === 0 ? (
                            <div className="py-20 text-center text-slate-550 italic">
                              No detections match the filters. Try adjusting search or confidence settings.
                            </div>
                          ) : (
                            <table className="w-full text-left border-collapse text-xs select-none">
                              <thead className="bg-slate-900/60 sticky top-0 z-20 border-b border-slate-800 text-slate-400 font-bold uppercase text-[10px] tracking-wider">
                                <tr>
                                  <th onClick={() => toggleReportSort('id')} className="py-3 px-4 cursor-pointer hover:bg-slate-800 transition-colors">
                                    ID{renderSortIcon('id')}
                                  </th>
                                  <th className="py-3 px-3">Symbol</th>
                                  <th onClick={() => toggleReportSort('yolo_conf')} className="py-3 px-3 cursor-pointer hover:bg-slate-800 transition-colors">
                                    YOLO{renderSortIcon('yolo_conf')}
                                  </th>
                                  <th onClick={() => toggleReportSort('class_conf')} className="py-3 px-3 cursor-pointer hover:bg-slate-800 transition-colors">
                                    Class{renderSortIcon('class_conf')}
                                  </th>
                                  <th onClick={() => toggleReportSort('pred_angle')} className="py-3 px-3 cursor-pointer hover:bg-slate-800 transition-colors">
                                    Angle{renderSortIcon('pred_angle')}
                                  </th>
                                  <th className="py-3 px-3">GT Exp</th>
                                  <th className="py-3 px-3">GT Ang</th>
                                  <th onClick={() => toggleReportSort('iou')} className="py-3 px-3 cursor-pointer hover:bg-slate-800 transition-colors">
                                    IoU{renderSortIcon('iou')}
                                  </th>
                                  <th className="py-3 px-3 text-center">Status</th>
                                  <th className="py-3 px-4 text-center">Action</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-900/40">
                                {sortedReportDetections.map((det) => {
                                  const isSelected = selectedReportRowId === det.id;
                                  const isHovered = hoveredReportBox === det.id;
                                  
                                  const hasGt = det.iou > 0.4;
                                  const isCorrect = hasGt && det.is_class_correct === true && det.is_orient_correct === true;
                                  const isClassWrong = hasGt && det.is_class_correct === false;
                                  const isOrientWrong = hasGt && det.is_orient_correct === false;
                                  
                                  // Color code rows based on status
                                  let rowClass = "hover:bg-slate-900/40 transition-colors cursor-pointer";
                                  if (isSelected) {
                                    rowClass = "bg-amber-950/20 border-l-2 border-amber-500 font-medium";
                                  } else if (isHovered) {
                                    rowClass = "bg-slate-900/60 font-medium";
                                  } else if (hasGt) {
                                    if (isCorrect) {
                                      rowClass = "bg-emerald-950/5 hover:bg-emerald-950/10 transition-colors";
                                    } else if (isClassWrong) {
                                      rowClass = "bg-red-950/5 hover:bg-red-950/10 transition-colors";
                                    } else if (isOrientWrong) {
                                      rowClass = "bg-amber-950/5 hover:bg-amber-950/10 transition-colors";
                                    }
                                  }

                                  return (
                                    <tr
                                      key={`table-row-${det.id}`}
                                      id={`report-row-${det.id}`}
                                      onClick={() => {
                                        setSelectedReportRowId(det.id);
                                      }}
                                      onMouseEnter={() => setHoveredReportBox(det.id)}
                                      onMouseLeave={() => setHoveredReportBox(null)}
                                      className={rowClass}
                                    >
                                      <td className="py-2.5 px-4 font-bold text-slate-400">#{det.id}</td>
                                      <td className="py-2.5 px-3">
                                        <span className="font-mono text-sm font-extrabold text-amber-400 bg-slate-950/80 px-2 py-0.5 rounded border border-slate-850">
                                          '{det.prediction}'
                                        </span>
                                      </td>
                                      <td className="py-2.5 px-3 font-semibold text-slate-300">{(det.yolo_conf * 100).toFixed(0)}%</td>
                                      <td className="py-2.5 px-3 font-semibold">
                                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                                          det.class_conf >= 0.8 ? 'bg-emerald-500/10 text-emerald-450' : 'bg-red-500/10 text-red-455'
                                        }`}>
                                          {(det.class_conf * 100).toFixed(0)}%
                                        </span>
                                      </td>
                                      <td className="py-2.5 px-3 font-mono text-slate-350">{det.pred_angle >= 0 ? '+' : ''}{det.pred_angle.toFixed(1)}°</td>
                                      <td className="py-2.5 px-3">
                                        {det.gt_expected ? (
                                          <span className="font-mono font-bold text-slate-400">'{det.gt_expected}'</span>
                                        ) : '-'}
                                      </td>
                                      <td className="py-2.5 px-3 font-mono text-slate-500">
                                        {det.gt_angle !== null ? `${det.gt_angle >= 0 ? '+' : ''}${det.gt_angle.toFixed(1)}°` : '-'}
                                      </td>
                                      <td className="py-2.5 px-3 font-semibold text-slate-400">{det.iou > 0 ? det.iou.toFixed(3) : '0.000'}</td>
                                      <td className="py-2.5 px-3 text-center">
                                        {hasGt ? (
                                          isCorrect ? (
                                            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/15 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                                              ✓ Match
                                            </span>
                                          ) : isClassWrong ? (
                                            <span className="text-[10px] font-bold text-red-400 bg-red-500/15 border border-red-500/20 px-2 py-0.5 rounded-full">
                                              ✗ Class
                                            </span>
                                          ) : (
                                            <span className="text-[10px] font-bold text-amber-400 bg-amber-500/15 border border-amber-500/20 px-2 py-0.5 rounded-full">
                                              ↻ Angle
                                            </span>
                                          )
                                        ) : (
                                          <span className="text-[10px] font-semibold text-slate-500 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded-full">
                                            Unlabeled
                                          </span>
                                        )}
                                      </td>
                                      <td className="py-2.5 px-4 text-center">
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            setSelectedReportRowId(det.id);
                                            handleSelectReportCrop(det);
                                          }}
                                          className="p-1 px-2.5 bg-slate-900 hover:bg-amber-500 hover:text-slate-950 border border-slate-800 hover:border-amber-400 rounded-md text-[10px] font-bold text-slate-300 transition-all flex items-center gap-1 mx-auto"
                                        >
                                          <Eye className="w-3 h-3" />
                                          Steps
                                        </button>
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          )}
                        </div>

                      </div>

                    </div>

                  </div>
                );
              })()}
            </div>

          </div>
        )}
      </div>

      {/* ── DETAIL JOURNEY MODAL: Shows the 5 steps of Preprocessing ── */}
      <AnimatePresence>
        {selectedCrop && (
          <div className="fixed inset-0 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-6 z-50 overflow-y-auto">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-slate-900 border border-slate-800 rounded-2xl max-w-6xl w-full flex flex-col overflow-hidden shadow-2xl relative"
            >
              {/* Modal Header */}
              <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between text-slate-100 bg-slate-950/25">
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-slate-550 uppercase tracking-widest">Detailed Preprocessing Journey</span>
                  <span className="text-base font-extrabold text-slate-100 mt-0.5">Box #{selectedCrop.id + 1} Pipeline Transformations</span>
                </div>
                <button
                  onClick={() => setSelectedCrop(null)}
                  className="px-3 py-1 bg-slate-850 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 text-xs font-bold rounded-lg transition-colors"
                >
                  Close Modal
                </button>
              </div>

              {/* Modal Body: 5 Preprocessing Steps with Synced Hover Magnifier */}
              <div className="p-6 flex flex-col gap-6 select-none">
                {selectedCrop.crops ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4 relative">
                    {renderCropStage(
                      1,
                      "Raw Crop",
                      selectedCrop.crops.raw,
                      "Original region of interest extracted directly from the source drawing layout.",
                      false // Dark background
                    )}

                    {renderCropStage(
                      2,
                      "Grayscale",
                      selectedCrop.crops.gray,
                      "Normalized single-channel intensity, removing color noise and standardizing brightness levels.",
                      true // Light background
                    )}

                    {renderCropStage(
                      3,
                      "Threshold",
                      selectedCrop.crops.thresh,
                      "Adaptive binarization converts pixels to pure black & white, separating ink from background.",
                      true
                    )}

                    {renderCropStage(
                      4,
                      "Line Removal",
                      selectedCrop.crops.clean,
                      "Morphological line subtraction filters out drawing grids and borders, leaving only the target symbol.",
                      true
                    )}

                    {renderCropStage(
                      5,
                      "Rectified",
                      selectedCrop.crops.rect,
                      "Deskews the crop by rotating it back to 0° alignment, cleaning final artifacts for classifier input.",
                      true
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-20 gap-4 border border-dashed border-slate-800 rounded-xl bg-slate-950">
                    <Loader2 className="w-10 h-10 animate-spin text-amber-500" />
                    <div className="text-center">
                      <span className="text-sm font-bold text-slate-200 block">Slicing Crops Dynamically</span>
                      <span className="text-xs text-slate-505 mt-1 block">The backend is running preprocessor stages on the original drawing file...</span>
                    </div>
                  </div>
                )}

                {/* Final Classification & Metadata Section */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-5 rounded-xl bg-slate-950 border border-slate-850">
                  <div className="flex flex-col gap-1.5">
                    <span className="text-[10px] font-bold text-slate-555 uppercase tracking-widest">MobileNetV3 Classifier Output</span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="font-mono text-3xl font-black text-amber-400">'{selectedCrop.pred_char}'</span>
                      {selectedCrop.pred_class && (
                        <span className="text-xs font-semibold text-slate-400">({selectedCrop.pred_class})</span>
                      )}
                    </div>
                    {/* Add Ground Truth match if available */}
                    {selectedCrop.best_iou > 0.4 ? (
                      <div className="flex items-center gap-1.5 mt-1">
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                          selectedCrop.is_class_correct 
                            ? 'bg-emerald-500/10 text-emerald-455 border-emerald-500/10' 
                            : 'bg-red-500/10 text-red-455 border-red-500/10'
                        }`}>
                          {selectedCrop.is_class_correct ? '✓ Correct Symbol Classification' : `✗ Mismatch: Expected '${selectedCrop.gt_expected_char}'`}
                        </span>
                      </div>
                    ) : (
                      <span className="text-[10px] text-slate-500 italic mt-1 bg-slate-900 border border-slate-850 px-2 py-1 rounded w-fit">
                        No Ground Truth annotations matched
                      </span>
                    )}
                  </div>

                  <div className="flex flex-col gap-1 border-t md:border-t-0 md:border-l border-slate-850 pt-3 md:pt-0 md:pl-6">
                    <span className="text-[10px] font-bold text-slate-555 uppercase tracking-widest">Model Confidences & IoU</span>
                    <div className="flex flex-col gap-1.5 mt-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400 font-medium">YOLO OBB Scan:</span>
                        <span className="text-slate-200 font-bold">
                          {selectedCrop.yolo_conf !== null && selectedCrop.yolo_conf !== undefined 
                            ? `${(selectedCrop.yolo_conf * 100).toFixed(1)}%` 
                            : 'N/A'
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400 font-medium">Character Classifier:</span>
                        <span className={`font-bold ${
                          selectedCrop.class_confidence !== null && selectedCrop.class_confidence !== undefined
                            ? selectedCrop.class_confidence >= 0.8 ? 'text-emerald-400' : 'text-red-400'
                            : 'text-slate-400'
                        }`}>
                          {selectedCrop.class_confidence !== null && selectedCrop.class_confidence !== undefined
                            ? `${(selectedCrop.class_confidence * 100).toFixed(1)}%` 
                            : 'N/A'
                          }
                        </span>
                      </div>
                      {selectedCrop.best_iou > 0.0 && (
                        <div className="flex justify-between border-t border-slate-900 pt-1">
                          <span className="text-slate-400 font-medium">Annotation IoU:</span>
                          <span className="text-blue-400 font-bold">{selectedCrop.best_iou.toFixed(3)}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex flex-col gap-1 border-t md:border-t-0 md:border-l border-slate-850 pt-3 md:pt-0 md:pl-6">
                    <span className="text-[10px] font-bold text-slate-555 uppercase tracking-widest">OBB Dimensions & Deskew</span>
                    <div className="flex flex-col gap-1.5 mt-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400 font-medium">Deskew Angle:</span>
                        <span className={`font-mono font-bold ${
                          selectedCrop.best_iou > 0.4 && selectedCrop.is_orient_correct === false ? 'text-amber-400' : 'text-slate-200'
                        }`}>
                          {selectedCrop.rotation_degrees !== null && selectedCrop.rotation_degrees !== undefined 
                            ? `${-selectedCrop.rotation_degrees.toFixed(1)}°` 
                            : '0°'
                          }
                          {selectedCrop.best_iou > 0.4 && selectedCrop.gt_angle !== null && selectedCrop.gt_angle !== undefined && (
                            <span className="text-[9px] font-semibold text-slate-400 ml-1.5">
                              (GT: {-selectedCrop.gt_angle.toFixed(1)}°)
                            </span>
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400 font-medium">Alignment Validation:</span>
                        <span className={`font-bold ${
                          selectedCrop.best_iou > 0.4 
                            ? selectedCrop.is_orient_correct 
                              ? 'text-emerald-400' 
                              : 'text-amber-400'
                            : 'text-slate-550'
                        }`}>
                          {selectedCrop.best_iou > 0.4 
                            ? selectedCrop.is_orient_correct 
                              ? '✓ Correct Angle' 
                              : '✗ Angle Mismatch'
                            : 'Unlabeled'
                          }
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

    </div>
  );
}

export default App;
