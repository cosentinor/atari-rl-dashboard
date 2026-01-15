/**
 * Atari RL Training Dashboard
 * Full-featured layout with all components
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import Icon from "@mui/material/Icon";
import Collapse from "@mui/material/Collapse";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import MDButton from "components/MDButton";
import DashboardLayout from "examples/LayoutContainers/DashboardLayout";
import DashboardNavbar from "examples/Navbars/DashboardNavbar";
import Footer from "examples/Footer";

// Import our custom components
import GameCanvas from "components/Atari/GameCanvas";
import ControlPanel from "components/Atari/ControlPanel";
import ModelExplanation from "components/Atari/ModelExplanation";
import StatsDisplay from "components/Atari/StatsDisplay";

// Import chart components
import RewardChart from "components/Atari/Charts/RewardChart";
import LossChart from "components/Atari/Charts/LossChart";
import QValueChart from "components/Atari/Charts/QValueChart";
import ActionChart from "components/Atari/Charts/ActionChart";
import RewardDistChart from "components/Atari/Charts/RewardDistChart";

// Import additional components
import EnhancedLeaderboard from "components/Atari/EnhancedLeaderboard";
import FeedbackWidget from "components/Atari/FeedbackWidget";
import ComparisonView from "components/Atari/ComparisonView";
import EmailModal from "components/Atari/EmailModal";

// Import services
import socketService from "services/socket.service";
import analyticsService from "services/analytics.service";
import config from "config";
import { PODCAST } from "constants/podcast";

const ACTION_MEANINGS = {
  3: ['NOOP', 'UP', 'DOWN'],
  4: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
  6: ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN'],
  9: ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT'],
  14: ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE'],
  18: ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'],
};

const DEFAULT_GAMES = [
  {
    id: 'ALE/Pong-v5',
    name: 'Pong',
    display_name: 'Pong',
    trained_episodes: null,
    action_space_size: 6,
    action_names: ACTION_MEANINGS[6],
  },
  {
    id: 'ALE/Breakout-v5',
    name: 'Breakout',
    display_name: 'Breakout',
    trained_episodes: null,
    action_space_size: 4,
    action_names: ACTION_MEANINGS[4],
  },
  {
    id: 'ALE/SpaceInvaders-v5',
    name: 'SpaceInvaders',
    display_name: 'Space Invaders',
    trained_episodes: null,
    action_space_size: 6,
    action_names: ACTION_MEANINGS[6],
  },
  {
    id: 'ALE/MsPacman-v5',
    name: 'MsPacman',
    display_name: 'Ms. Pac-Man',
    trained_episodes: null,
    action_space_size: 9,
    action_names: ACTION_MEANINGS[9],
  },
  {
    id: 'ALE/Asteroids-v5',
    name: 'Asteroids',
    display_name: 'Asteroids',
    trained_episodes: null,
    action_space_size: 14,
    action_names: ACTION_MEANINGS[14],
  },
  {
    id: 'ALE/Boxing-v5',
    name: 'Boxing',
    display_name: 'Boxing',
    trained_episodes: null,
    action_space_size: 18,
    action_names: ACTION_MEANINGS[18],
  },
  {
    id: 'ALE/BeamRider-v5',
    name: 'BeamRider',
    display_name: 'Beam Rider',
    trained_episodes: null,
    action_space_size: 9,
    action_names: ACTION_MEANINGS[9],
  },
  {
    id: 'ALE/Seaquest-v5',
    name: 'Seaquest',
    display_name: 'Seaquest',
    trained_episodes: null,
    action_space_size: 18,
    action_names: ACTION_MEANINGS[18],
  },
  {
    id: 'ALE/Enduro-v5',
    name: 'Enduro',
    display_name: 'Enduro',
    trained_episodes: null,
    action_space_size: 9,
    action_names: ACTION_MEANINGS[9],
  },
  {
    id: 'ALE/Freeway-v5',
    name: 'Freeway',
    display_name: 'Freeway',
    trained_episodes: null,
    action_space_size: 3,
    action_names: ACTION_MEANINGS[3],
  },
];

function AtariDashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [games, setGames] = useState(DEFAULT_GAMES);
  const [selectedGame, setSelectedGame] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [savedModels, setSavedModels] = useState([]);
  const [loadCheckpoint, setLoadCheckpoint] = useState('');
  const [resumeFromSaved, setResumeFromSaved] = useState(false);
  const [checkpointRefreshKey, setCheckpointRefreshKey] = useState(0);
  const [availableCheckpoints, setAvailableCheckpoints] = useState([]);
  const [pretrainedLevels, setPretrainedLevels] = useState({ low: null, medium: null, high: null });
  const [pretrainedLoading, setPretrainedLoading] = useState(false);
  const [trainingSpeed, setTrainingSpeed] = useState('1x');
  const [trainingLevel, setTrainingLevel] = useState('medium');
  
  // Stats
  const [stats, setStats] = useState({
    episode: 0,
    reward: 0,
    bestReward: 0,
    loss: 0,
    qValue: 0,
    fps: 0,
    steps: 0
  });
  
  // Chart data
  const [episodes, setEpisodes] = useState([]);
  const [actionDist, setActionDist] = useState({});
  const [rewardDist, setRewardDist] = useState({ bins: [], counts: [] });
  const [numActions, setNumActions] = useState(6);
  const [actionNames, setActionNames] = useState([]);
  
  // Logs
  const [logs, setLogs] = useState([]);
  const [logExpanded, setLogExpanded] = useState(false);
  
  // Modals
  const [showComparison, setShowComparison] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [podcastLogoError, setPodcastLogoError] = useState(false);
  
  // Chart tabs
  const [chartTab, setChartTab] = useState(0);
  
  const addLog = useCallback((message, type = 'info') => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-50), { time, message, type }]);
  }, []);
  
  // Initialize analytics
  useEffect(() => {
    analyticsService.initialize();
    return () => analyticsService.cleanup();
  }, []);
  
  // Initialize WebSocket connection
  useEffect(() => {
    console.log('Connecting to backend...');
    const socket = socketService.connect();
    
    // Connection events
    socket.on('connect', () => {
      console.log('✓ Connected to backend');
      setIsConnected(true);
      addLog('Connected to server', 'success');
      socket.emit(config.events.getInit);
    });
    
    socket.on('disconnect', () => {
      console.log('✗ Disconnected from backend');
      setIsConnected(false);
      setIsTraining(false);
      addLog('Disconnected from server', 'error');
    });
    
    // Initialize data
    socket.on(config.events.init, (data) => {
      console.log('Received init data:', data);
      const gameMap = new Map(DEFAULT_GAMES.map((game) => [game.id, game]));
      if (data.games && data.games.length > 0) {
        data.games.forEach((game) => {
          gameMap.set(game.id, { ...gameMap.get(game.id), ...game });
        });
      }
      setGames(Array.from(gameMap.values()));
      setIsTraining(data.isTraining || false);
      setSavedModels(data.savedModels || []);
      setTrainingSpeed(data.trainingSpeed || '1x');
      setTrainingLevel(data.trainingLevel || 'medium');
      addLog(`Loaded ${data.games?.length || 0} games`);
    });
    
    // Frame updates
    socket.on(config.events.frame, (data) => {
      if (window.addFrame && data.data) {
        window.addFrame(data.data);
      }
      
      setStats(prev => ({
        episode: data.episode || prev.episode,
        reward: data.reward || 0,
        bestReward: Math.max(prev.bestReward, data.reward || 0),
        loss: data.loss?.toFixed(4) || prev.loss,
        qValue: data.qValue?.toFixed(2) || prev.qValue,
        fps: data.fps || prev.fps,
        steps: data.step || prev.steps
      }));
    });
    
    // Training events
    socket.on(config.events.trainingStarted, (data) => {
      console.log('Training started:', data);
      setIsTraining(true);
      setSessionId(data.sessionId);
      setStats(prev => ({ ...prev, bestReward: 0 }));
      if (window.clearCanvas) {
        window.clearCanvas();
      }
      if (data.trainingLevel) {
        setTrainingLevel(data.trainingLevel);
      }
      addLog(`Training started: ${data.game}`, 'success');
    });
    
    socket.on(config.events.trainingStopped, () => {
      console.log('Training stopped');
      setIsTraining(false);
      setCheckpointRefreshKey((prev) => prev + 1);
      addLog('Training stopped');
    });

    socket.on(config.events.status, (data) => {
      if (typeof data.isTraining === 'boolean') {
        setIsTraining(data.isTraining);
      }
      if (!data.isTraining) {
        setSessionId(null);
      }
    });
    
    socket.on(config.events.episodeEnd, (data) => {
      console.log('Episode ended:', data);
      
      // Update episodes for charts
      setEpisodes(prev => {
        const newEp = {
          episode_num: data.episode,
          episode: data.episode,
          reward: data.reward,
          steps: data.steps,
          loss: data.loss,
          q_value_mean: data.qValueMean,
          q_value_max: data.qValueMax
        };
        return [...prev, newEp].slice(-200);
      });
      
      setStats(prev => ({
        ...prev,
        bestReward: data.bestReward || prev.bestReward
      }));
      addLog(`Episode ${data.episode}: reward=${data.reward?.toFixed(1)}, steps=${data.steps}`);

      if (data.actionDistribution) {
        setActionDist(data.actionDistribution);
      }
      
      // Show email modal after first episode
      if (data.episode === 1) {
        const emailStatus = localStorage.getItem('email_collected');
        if (!emailStatus) {
          setShowEmailModal(true);
        }
      }
    });
    
    // Model events
    socket.on(config.events.modelSaved, (data) => {
      addLog(`Model saved at episode ${data.episode}`, 'success');
      setCheckpointRefreshKey((prev) => prev + 1);
    });
    
    socket.on(config.events.modelLoaded, (data) => {
      addLog(`Loaded: ${data.checkpoint}`, 'success');
    });
    
    // Speed change
    socket.on(config.events.speedChanged, (data) => {
      if (data.trainingSpeed) setTrainingSpeed(data.trainingSpeed);
    });
    
    // Logs
    socket.on(config.events.log, (data) => {
      addLog(data.message, data.type || 'info');
    });
    
    // Errors
    socket.on(config.events.error, (data) => {
      addLog(`Error: ${data.message}`, 'error');
    });
    
    // History data for charts
    socket.on('history_data', (data) => {
      if (data.episodes) setEpisodes(data.episodes);
      if (data.actionDistribution) setActionDist(data.actionDistribution);
      if (data.rewardDistribution) setRewardDist(data.rewardDistribution);
    });
    
    return () => {
      console.log('Disconnecting...');
      socketService.disconnect();
    };
  }, [addLog]);
  
  // Handle game change
  const handleGameChange = useCallback((newGame) => {
    if (isTraining) {
      addLog('Stopping training to switch games...', 'warning');
      socketService.emit(config.events.stopTraining);
    }

    setSelectedGame(newGame);
    setLoadCheckpoint('');
    setResumeFromSaved(false);
    setAvailableCheckpoints([]);
    setPretrainedLevels({ low: null, medium: null, high: null });
    setSessionId(null);
    setStats({
      episode: 0,
      reward: 0,
      bestReward: 0,
      loss: 0,
      qValue: 0,
      fps: 0,
      steps: 0
    });
    
    setEpisodes([]);
    setActionDist({});
    setRewardDist({ bins: [], counts: [] });
    
    const gameInfo = games.find(g => g.id === newGame);
    if (gameInfo && gameInfo.action_names) {
      setActionNames(gameInfo.action_names);
      setNumActions(gameInfo.action_space_size);
    } else {
      setActionNames([]);
      setNumActions(6);
    }
    
    if (window.clearCanvas) {
      window.clearCanvas();
    }
    
    addLog(`Selected game: ${newGame}`);
  }, [isTraining, addLog, games]);

  const handleCheckpointsLoaded = useCallback((checkpointList) => {
    const next = Array.isArray(checkpointList) ? checkpointList : [];
    setAvailableCheckpoints(next);
  }, []);

  const selectedGameInfo = useMemo(
    () => games.find((game) => game.id === selectedGame) || null,
    [games, selectedGame]
  );

  useEffect(() => {
    if (!selectedGame) {
      setPretrainedLevels({ low: null, medium: null, high: null });
      return;
    }
    const gameKey = selectedGame.replace(/\//g, '_');
    setPretrainedLoading(true);
    fetch(`${config.API_BASE_URL}/api/pretrained/${gameKey}`)
      .then(r => r.json())
      .then(data => {
        if (data && data.success && data.levels) {
          setPretrainedLevels(data.levels);
        } else {
          setPretrainedLevels({ low: null, medium: null, high: null });
        }
      })
      .catch(() => {
        setPretrainedLevels({ low: null, medium: null, high: null });
      })
      .finally(() => {
        setPretrainedLoading(false);
      });
  }, [selectedGame]);

  const hasPretrainedModels = useMemo(
    () => Boolean(pretrainedLevels.low || pretrainedLevels.medium || pretrainedLevels.high),
    [pretrainedLevels]
  );

  const selectedPretrainedModel = useMemo(
    () => pretrainedLevels[trainingLevel] || null,
    [pretrainedLevels, trainingLevel]
  );

  useEffect(() => {
    const source = (selectedPretrainedModel?.source || '').toLowerCase();
    const isLocalPretrained = source === 'local' || source === 'rc_model';

    if (selectedPretrainedModel && isLocalPretrained) {
      if (selectedPretrainedModel.filename) {
        setResumeFromSaved(true);
        setLoadCheckpoint(selectedPretrainedModel.filename);
      } else {
        setResumeFromSaved(false);
        setLoadCheckpoint('');
      }
      return;
    }

    if (selectedPretrainedModel && !isLocalPretrained) {
      if (Array.isArray(availableCheckpoints) && availableCheckpoints.length > 0) {
        const sorted = [...availableCheckpoints].sort(
          (a, b) => (a.episode ?? 0) - (b.episode ?? 0)
        );
        const latest = sorted[sorted.length - 1];
        if (latest && latest.filename) {
          setResumeFromSaved(true);
          setLoadCheckpoint(latest.filename);
          return;
        }
      }
      setResumeFromSaved(false);
      setLoadCheckpoint('');
      return;
    }

    if (Array.isArray(availableCheckpoints) && availableCheckpoints.length > 0) {
      const sorted = [...availableCheckpoints].sort(
        (a, b) => (a.episode ?? 0) - (b.episode ?? 0)
      );
      const latest = sorted[sorted.length - 1];
      if (latest && latest.filename) {
        setResumeFromSaved(true);
        setLoadCheckpoint(latest.filename);
        return;
      }
    }
    setResumeFromSaved(false);
    setLoadCheckpoint('');
  }, [selectedPretrainedModel, availableCheckpoints]);

  const handleDownloadWeights = useCallback((gameId, checkpointName, pretrainedId) => {
    if (!gameId) return;
    const baseUrl = config.API_BASE_URL || '';

    if (pretrainedId) {
      const url = `${baseUrl}/api/pretrained/download?id=${encodeURIComponent(pretrainedId)}`;
      window.open(url, '_blank', 'noopener,noreferrer');
      return;
    }

    const gameKey = gameId.replace(/\//g, '_');
    let url = `${baseUrl}/api/models/${gameKey}/download`;

    if (checkpointName) {
      url += `?mode=checkpoint&checkpoint=${encodeURIComponent(checkpointName)}`;
    } else {
      url += '?mode=latest';
    }

    window.open(url, '_blank', 'noopener,noreferrer');
  }, []);
  
  // Start training
  const handleStart = useCallback(() => {
    if (selectedGame) {
      const source = (selectedPretrainedModel?.source || '').toLowerCase();
      const isLocalPretrained = ['local', 'rc_model'].includes(source);
      const shouldWarmStartPretrained = Boolean(
        selectedPretrainedModel && !isLocalPretrained && !resumeFromSaved
      );
      const runMode = 'train';
      const logLabel = shouldWarmStartPretrained
        ? 'Starting training from pre-trained weights'
        : (resumeFromSaved || isLocalPretrained ? 'Resuming training' : 'Starting training');
      addLog(`${logLabel} for ${selectedGame}...`);
      analyticsService.trackTrainingStart(selectedGame);

      const payload = {
        game: selectedGame,
        trainingLevel,
        runMode,
      };

      if (isLocalPretrained && selectedPretrainedModel?.filename) {
        payload.loadCheckpoint = selectedPretrainedModel.filename;
        payload.resumeFromSaved = true;
      } else if (shouldWarmStartPretrained) {
        payload.pretrainedId = selectedPretrainedModel?.id || null;
        payload.loadCheckpoint = null;
        payload.resumeFromSaved = false;
      } else {
        payload.loadCheckpoint = resumeFromSaved ? (loadCheckpoint || null) : null;
        payload.resumeFromSaved = resumeFromSaved;
      }

      socketService.emit(config.events.startTraining, payload);
    }
  }, [selectedGame, loadCheckpoint, resumeFromSaved, trainingLevel, selectedPretrainedModel, addLog]);
  
  // Stop training
  const handleStop = useCallback(() => {
    addLog('Stopping training...');
    socketService.emit(config.events.stopTraining);
    fetch(`${config.API_BASE_URL}/api/training/stop`, { method: 'POST' }).catch(() => {});
    // Optimistically update UI in case backend takes time to emit stop event
    setIsTraining(false);
    if (window.clearCanvas) {
      window.clearCanvas();
    }
    setStats({
      episode: 0,
      reward: 0,
      bestReward: 0,
      loss: 0,
      qValue: 0,
      fps: 0,
      steps: 0
    });
    setEpisodes([]);
    setActionDist({});
    setRewardDist({ bins: [], counts: [] });
    setSessionId(null);
  }, [addLog]);
  
  // Save model
  const handleSave = useCallback(() => {
    addLog('Saving model...');
    socketService.emit(config.events.saveModel);
  }, [addLog]);
  
  // Delete checkpoint
  const handleDeleteCheckpoint = useCallback((gameId, checkpointName) => {
    addLog(`Deleting checkpoint: ${checkpointName}...`);
    socketService.emit(config.events.deleteCheckpoint, {
      game_id: gameId,
      checkpoint: checkpointName
    });
  }, [addLog]);
  
  // Training level change
  const handleTrainingLevelChange = useCallback((level) => {
    setTrainingLevel(level);
    if (selectedGame) {
      socketService.emit(config.events.setTrainingLevel, {
        trainingLevel: level,
        game: selectedGame,
      });
    }
    addLog(`Training level set to ${level}`);
  }, [addLog, selectedGame]);

  // Training speed change
  const handleTrainingSpeedChange = useCallback((speed) => {
    socketService.emit(config.events.setTrainingSpeed, { speed });
    analyticsService.trackSpeedChange(speed);
    addLog(`Speed changed to ${speed}`);
  }, [addLog]);
  
  // Autosave every 90 seconds during training
  useEffect(() => {
    if (!isTraining || !sessionId) return;
    
    const interval = setInterval(() => {
      socketService.emit(config.events.saveModel);
      addLog('✓ Auto-saved model', 'success');
    }, 90000);
    
    return () => clearInterval(interval);
  }, [isTraining, sessionId, addLog]);
  
  // Autosave when user leaves page
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (isTraining) {
        socketService.emit(config.events.saveModel);
        e.preventDefault();
        e.returnValue = 'Training in progress. Your model will be saved.';
        return e.returnValue;
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isTraining]);

  const hasEpisodes = useMemo(() => episodes.length > 0, [episodes]);
  const hasActionDist = useMemo(() => actionDist && Object.keys(actionDist).length > 0, [actionDist]);
  const hasRewardDist = useMemo(
    () => rewardDist?.bins && rewardDist.bins.length > 0 && rewardDist.counts?.length > 0,
    [rewardDist]
  );

  // Fetch history (episodes + distributions) when session is known
  useEffect(() => {
    if (!sessionId) return;
    socketService.emit(config.events.getHistory, { sessionId });
  }, [sessionId]);


  return (
    <DashboardLayout>
      <DashboardNavbar />
      <MDBox py={2}>
        {/* Header Row */}
        <MDBox mb={2}>
          <Card
            sx={{
              borderRadius: '20px',
              background: 'linear-gradient(135deg, #131a2c 0%, #0d1424 100%)',
              border: '1px solid rgba(148, 163, 184, 0.25)',
              boxShadow: '0 20px 40px rgba(0,0,0,0.45)',
            }}
          >
            <MDBox
              px={{ xs: 2, md: 3 }}
              py={{ xs: 2, md: 2.5 }}
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              gap={2}
            >
              <MDBox
                display="flex"
                alignItems="center"
                gap={2}
                sx={{ flexWrap: { xs: 'wrap', md: 'nowrap' }, rowGap: 1.5 }}
              >
                <MDBox
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  width="44px"
                  height="44px"
                  borderRadius="12px"
                  sx={{
                    background: 'linear-gradient(180deg, #0ea5e9 0%, #8b5cf6 100%)',
                    color: '#0b1020',
                  }}
                >
                  <Icon sx={{ color: 'white', fontSize: '1.8rem !important' }}>sports_esports</Icon>
                </MDBox>
                <MDBox>
                  <MDTypography
                    variant="h4"
                    fontWeight="bold"
                    sx={{
                      background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      lineHeight: 1.1,
                    }}
                  >
                    Atari RL Training
                  </MDTypography>
                  <MDBox display="flex" alignItems="center" gap={1}>
                    <MDTypography
                      variant="button"
                      sx={{
                        color: 'rgba(226, 232, 240, 0.9)',
                        fontWeight: 500,
                        borderBottom: '1px dotted rgba(148, 163, 184, 0.5)',
                      }}
                    >
                      Rainbow DQN Dashboard
                    </MDTypography>
                    <Tooltip
                      title="Monitor training, metrics, and controls for Rainbow DQN agents."
                      arrow
                      componentsProps={{
                        tooltip: {
                          sx: {
                            backgroundColor: 'rgba(15, 23, 42, 0.82)',
                            border: '1px solid rgba(148, 163, 184, 0.3)',
                            color: '#e2e8f0',
                            fontSize: '0.75rem',
                            boxShadow: '0 12px 24px rgba(0,0,0,0.35)',
                          },
                        },
                        arrow: { sx: { color: 'rgba(15, 23, 42, 0.82)' } },
                      }}
                    >
                      <Icon fontSize="small" sx={{ color: 'rgba(226,232,240,0.7)' }}>
                        info
                      </Icon>
                    </Tooltip>
                  </MDBox>
                </MDBox>
                <MDBox display="flex" alignItems="center" gap={1.5}>
                  <MDBox
                    component="a"
                    href={PODCAST.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    display="inline-flex"
                    alignItems="center"
                    gap={1.25}
                    sx={{
                      textDecoration: 'none',
                      color: 'inherit',
                    }}
                    aria-label={`Subscribe to ${PODCAST.name} podcast`}
                  >
                    <MDBox
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      width="48px"
                      height="48px"
                      borderRadius="12px"
                      sx={{
                        backgroundColor: '#0b1224',
                        border: '1px solid rgba(148, 163, 184, 0.35)',
                        overflow: 'hidden',
                        textDecoration: 'none',
                      }}
                    >
                      {!podcastLogoError ? (
                        <img
                          src={PODCAST.logoSrc}
                          alt={PODCAST.logoAlt}
                          onError={() => setPodcastLogoError(true)}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                      ) : (
                        <MDTypography variant="caption" fontWeight="bold" sx={{ color: '#e2e8f0' }}>
                          {PODCAST.shortName}
                        </MDTypography>
                    )}
                  </MDBox>
                  <MDBox display="flex" flexDirection="column">
                    <MDTypography
                      variant="h4"
                      fontWeight="bold"
                      sx={{
                        background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        lineHeight: 1.05,
                      }}
                    >
                      Major Programmes
                    </MDTypography>
                    <MDTypography
                      variant="h4"
                      fontWeight="bold"
                      sx={{
                        background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        lineHeight: 1.05,
                      }}
                    >
                      Navigating
                    </MDTypography>
                  </MDBox>
                </MDBox>
                <MDButton
                  component="a"
                  href={PODCAST.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  variant="contained"
                  color="info"
                  size="small"
                  sx={{
                    textTransform: 'none',
                    borderRadius: '12px',
                    width: '120px',
                    height: '48px',
                    minWidth: '120px',
                    minHeight: '48px',
                    p: 0,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'linear-gradient(180deg, #0ea5e9 0%, #8b5cf6 100%)',
                    color: '#fff',
                    border: '1px solid rgba(14, 165, 233, 0.35)',
                    boxShadow: '0 10px 20px rgba(14, 165, 233, 0.25)',
                    '&:hover': {
                      background: 'linear-gradient(180deg, #22d3ee 0%, #6366f1 100%)',
                    },
                  }}
                >
                  Subscribe
                </MDButton>
                </MDBox>
              </MDBox>

              <MDBox display="flex" alignItems="center" gap={2}>
                <MDBox
                  display="flex"
                  alignItems="center"
                  gap={1}
                  px={2}
                  py={1}
                  borderRadius="24px"
                  sx={{
                    backgroundColor: isConnected ? 'rgba(22, 163, 74, 0.18)' : 'rgba(239, 68, 68, 0.18)',
                    border: `1px solid ${isConnected ? 'rgba(34, 197, 94, 0.45)' : 'rgba(248, 113, 113, 0.45)'}`,
                    color: isConnected ? '#34d399' : '#f87171',
                    minWidth: '160px',
                    justifyContent: 'center',
                  }}
                >
                  <span style={{ fontSize: '0.9rem' }}>●</span>
                  <MDTypography variant="button" fontWeight="medium">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </MDTypography>
                </MDBox>
                <MDBox display="flex" alignItems="center" gap={1} color="text">
                  <Icon sx={{ fontSize: '1.1rem !important', color: 'rgba(226,232,240,0.7)' }}>
                    desktop_windows
                  </Icon>
                  <MDTypography variant="button" color="text">
                    cpu
                  </MDTypography>
                </MDBox>
              </MDBox>
            </MDBox>
          </Card>
        </MDBox>

        {/* Row 1: Live Game + Controls/Stats */}
        <Grid container spacing={2} mb={2} alignItems="flex-start">
          <Grid item xs={12} lg={8}>
            <GameCanvas
              key={`${selectedGame || 'no-game'}-${sessionId || 'idle'}`}
              isTraining={isTraining}
              sessionId={sessionId}
              selectedGame={selectedGame}
              stats={stats}
            />
          </Grid>
          <Grid item xs={12} lg={4}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <ControlPanel
                  games={games}
                  selectedGame={selectedGame}
                  onGameChange={handleGameChange}
                  isTraining={isTraining}
                  onStart={handleStart}
                  onStop={handleStop}
                  onSave={handleSave}
                  savedModels={savedModels}
                  loadCheckpoint={loadCheckpoint}
                  resumeFromSaved={resumeFromSaved}
                  onLoadCheckpointChange={setLoadCheckpoint}
                  onResumeFromSavedChange={setResumeFromSaved}
                  onDownloadWeights={handleDownloadWeights}
                  pretrainedModel={selectedPretrainedModel}
                  pretrainedLoading={pretrainedLoading}
                  checkpointRefreshKey={checkpointRefreshKey}
                  onCheckpointsLoaded={handleCheckpointsLoaded}
                  onDeleteCheckpoint={handleDeleteCheckpoint}
                  trainingSpeed={trainingSpeed}
                  onTrainingSpeedChange={handleTrainingSpeedChange}
                  trainingLevel={trainingLevel}
                  onTrainingLevelChange={handleTrainingLevelChange}
                  hasPretrainedModels={hasPretrainedModels}
                />
              </Grid>
              <Grid item xs={12}>
                <StatsDisplay stats={stats} />
              </Grid>
            </Grid>
          </Grid>
        </Grid>

        {/* Row 2: Metrics + Leaderboard */}
        <Grid container spacing={2} mb={2} alignItems="stretch">
          <Grid item xs={12} lg={8}>
            <Card
              sx={{
                background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
                border: '1px solid rgba(148, 163, 184, 0.18)',
                boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
                borderRadius: '16px',
                height: '100%',
              }}
            >
              <MDBox p={3}>
                <MDBox display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <MDBox display="flex" alignItems="center" gap={1.5}>
                    <Icon sx={{ color: '#0ea5e9' }}>stacked_line_chart</Icon>
                    <MDTypography variant="h6" fontWeight="medium">
                      Metrics
                    </MDTypography>
                  </MDBox>
                </MDBox>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <RewardChart episodes={episodes} />
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <LossChart episodes={episodes} />
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <QValueChart episodes={episodes} />
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <ActionChart 
                      actionDist={actionDist} 
                      numActions={numActions}
                      actionNames={actionNames}
                    />
                  </Grid>
                </Grid>
              </MDBox>
            </Card>
          </Grid>

          <Grid item xs={12} lg={4}>
            <MDBox sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
              <ModelExplanation
                selectedGame={selectedGame}
                gameInfo={selectedGameInfo}
                trainingLevel={trainingLevel}
                pretrainedModel={selectedPretrainedModel}
                hasPretrainedModels={hasPretrainedModels}
              />
              <MDBox sx={{ flex: 1, minHeight: 0 }}>
                <EnhancedLeaderboard currentGame={selectedGame} />
              </MDBox>
            </MDBox>
          </Grid>
        </Grid>
        
        {/* Compare Models Button */}
        {selectedGame && (
          <MDBox mb={2}>
            <MDButton
              variant="outlined"
              color="info"
              onClick={() => setShowComparison(true)}
              startIcon={<Icon>compare_arrows</Icon>}
            >
              Compare Models
            </MDButton>
          </MDBox>
        )}

        {/* Activity Log - Collapsible at Bottom */}
        <MDBox mt={3}>
          <Card
            sx={{
              background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
              border: '1px solid rgba(148, 163, 184, 0.18)',
              boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
              borderRadius: '16px',
            }}
          >
            <MDBox 
              p={2} 
              display="flex" 
              justifyContent="space-between" 
              alignItems="center"
              sx={{ cursor: 'pointer' }}
              onClick={() => setLogExpanded(!logExpanded)}
            >
              <MDBox display="flex" alignItems="center" gap={1}>
                <Icon sx={{ color: '#0ea5e9' }}>terminal</Icon>
                <MDTypography variant="h6" fontWeight="medium">
                  Activity Log
                </MDTypography>
                <MDTypography variant="caption" color="text" ml={2}>
                  {logs.length} entries
                </MDTypography>
              </MDBox>
              <MDBox display="flex" alignItems="center" gap={2}>
                {/* Show last log entry when collapsed */}
                {!logExpanded && logs.length > 0 && (
                  <MDTypography 
                    variant="caption" 
                    color="text"
                    sx={{ 
                      fontFamily: 'monospace',
                      maxWidth: '400px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    [{logs[logs.length - 1]?.time}] {logs[logs.length - 1]?.message}
                  </MDTypography>
                )}
                <IconButton size="small">
                  <Icon>{logExpanded ? 'expand_less' : 'expand_more'}</Icon>
                </IconButton>
              </MDBox>
            </MDBox>
            
            <Collapse in={logExpanded}>
              <MDBox 
                px={2} 
                pb={2}
                sx={{ 
                  maxHeight: '200px', 
                  overflowY: 'auto',
                  '&::-webkit-scrollbar': {
                    width: '6px',
                  },
                  '&::-webkit-scrollbar-track': {
                    background: 'rgba(255,255,255,0.03)',
                    borderRadius: '3px',
                  },
                  '&::-webkit-scrollbar-thumb': {
                    background: '#06b6d4',
                    borderRadius: '3px',
                  },
                }}
              >
                {logs.slice(-50).reverse().map((log, i) => (
                  <MDTypography
                    key={i}
                    variant="caption"
                    display="block"
                    sx={{
                      color: log.type === 'error' ? 'error.main' : 
                             log.type === 'success' ? 'success.main' :
                             log.type === 'warning' ? 'warning.main' : 'rgba(226, 232, 240, 0.8)',
                      fontFamily: 'monospace',
                      fontSize: '0.75rem',
                      lineHeight: 1.5,
                      mb: 0.5,
                    }}
                  >
                    <span style={{ opacity: 0.6 }}>[{log.time}]</span> {log.message}
                  </MDTypography>
                ))}
              </MDBox>
            </Collapse>
          </Card>
        </MDBox>
      </MDBox>
      
      {/* Feedback Widget (Floating FAB) */}
      <FeedbackWidget />
      
      {/* Comparison View Modal */}
      <ComparisonView
        open={showComparison}
        onClose={() => setShowComparison(false)}
        gameId={selectedGame}
      />
      
      {/* Email Collection Modal */}
      <EmailModal
        open={showEmailModal}
        onClose={() => setShowEmailModal(false)}
      />
      
      <Footer />
    </DashboardLayout>
  );
}

export default AtariDashboard;