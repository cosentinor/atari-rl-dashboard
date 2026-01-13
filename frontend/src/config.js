/**
 * API Configuration for Material Dashboard PRO Integration
 * Place this file in: frontend/src/config.js
 */

const resolveBaseUrl = (envUrl) => {
  if (typeof window !== "undefined" && window.location) {
    const { protocol, hostname, port, origin } = window.location;
    const isLocalhost = hostname === "localhost" || hostname === "127.0.0.1";
    if (isLocalhost && port === "5001") {
      return origin;
    }

    if (!isLocalhost) {
      if (!envUrl || /localhost|127\.0\.0\.1/.test(envUrl)) {
        return `${protocol}//${hostname}:5001`;
      }
      return envUrl;
    }

    return envUrl || `${protocol}//${hostname}:5001`;
  }
  return envUrl || "http://localhost:5001";
};

const config = {
  // API Base URL - automatically switches based on environment
  API_BASE_URL: resolveBaseUrl(process.env.REACT_APP_API_URL),
  
  // WebSocket URL for real-time updates
  WS_URL: resolveBaseUrl(process.env.REACT_APP_WS_URL),
  
  // API Endpoints
  endpoints: {
    games: '/api/games',
    models: '/api/models',
    leaderboard: '/api/leaderboard',
    feedback: '/api/feedback',
    analytics: '/api/analytics'
  },
  
  // WebSocket Events
  events: {
    // Client -> Server
    startTraining: 'start_training',
    stopTraining: 'stop_training',
    saveModel: 'save_model',
    setTrainingSpeed: 'set_training_speed',
    setTrainingLevel: 'set_training_level',
    setVizSpeed: 'set_viz_speed',
    deleteCheckpoint: 'delete_checkpoint',
    getHistory: 'get_history',
    getInit: 'get_init',
    
    // Server -> Client
    init: 'init',
    frame: 'frame',
    trainingStarted: 'training_started',
    trainingStopped: 'training_stopped',
    episodeEnd: 'episode_end',
    modelSaved: 'model_saved',
    modelLoaded: 'model_loaded',
    historyData: 'history_data',
    speedChanged: 'speed_changed',
    status: 'status',
    log: 'log',
    error: 'error'
  },
  
  // App Settings
  settings: {
    maxLogEntries: 50,
    chartMaxDataPoints: 200,
    frameBufferSize: 30,
    canvasWidth: 420,
    canvasHeight: 320
  }
};

export default config;
