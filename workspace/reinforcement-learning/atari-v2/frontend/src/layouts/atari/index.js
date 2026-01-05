/**
 * Atari RL Training Dashboard
 * Full-featured layout with all components
 */

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
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
import StatsDisplay from "components/Atari/StatsDisplay";

// Import chart components
import RewardChart from "components/Atari/Charts/RewardChart";
import LossChart from "components/Atari/Charts/LossChart";
import QValueChart from "components/Atari/Charts/QValueChart";
import ActionChart from "components/Atari/Charts/ActionChart";
import RewardDistChart from "components/Atari/Charts/RewardDistChart";

// Import additional components
import EnhancedLeaderboard from "components/Atari/EnhancedLeaderboard";
import ChallengesPanel from "components/Atari/ChallengesPanel";
import FeedbackWidget from "components/Atari/FeedbackWidget";
import ComparisonView from "components/Atari/ComparisonView";
import EmailModal from "components/Atari/EmailModal";

// Import services
import socketService from "services/socket.service";
import analyticsService from "services/analytics.service";
import config from "config";

const DEFAULT_GAMES = [
  { id: 'Pong', name: 'Pong', display_name: 'Pong', trained_episodes: null },
  { id: 'Breakout', name: 'Breakout', display_name: 'Breakout', trained_episodes: null },
  { id: 'SpaceInvaders', name: 'SpaceInvaders', display_name: 'Space Invaders', trained_episodes: null },
  { id: 'MsPacman', name: 'MsPacman', display_name: 'Ms. Pac-Man', trained_episodes: null },
  { id: 'Asteroids', name: 'Asteroids', display_name: 'Asteroids', trained_episodes: null },
  { id: 'Boxing', name: 'Boxing', display_name: 'Boxing', trained_episodes: null },
  { id: 'BeamRider', name: 'BeamRider', display_name: 'Beam Rider', trained_episodes: null },
  { id: 'Seaquest', name: 'Seaquest', display_name: 'Seaquest', trained_episodes: null },
  { id: 'Enduro', name: 'Enduro', display_name: 'Enduro', trained_episodes: null },
  { id: 'Freeway', name: 'Freeway', display_name: 'Freeway', trained_episodes: null },
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
  const [trainingSpeed, setTrainingSpeed] = useState('1x');
  const resumeDefaultAppliedRef = useRef(false);
  
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
      setGames((data.games && data.games.length > 0) ? data.games : DEFAULT_GAMES);
      setIsTraining(data.isTraining || false);
      setSavedModels(data.savedModels || []);
      setTrainingSpeed(data.trainingSpeed || '1x');
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
      addLog(`Training started: ${data.game}`, 'success');
    });
    
    socket.on(config.events.trainingStopped, () => {
      console.log('Training stopped');
      setIsTraining(false);
      setCheckpointRefreshKey((prev) => prev + 1);
      addLog('Training stopped');
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
      addLog('Cannot change game while training', 'warning');
      return;
    }

    setSelectedGame(newGame);
    setLoadCheckpoint('');
    setResumeFromSaved(false);
    resumeDefaultAppliedRef.current = false;
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
    const hasAny = Array.isArray(checkpointList) && checkpointList.length > 0;
    if (!resumeDefaultAppliedRef.current) {
      setResumeFromSaved(hasAny);
      resumeDefaultAppliedRef.current = true;
    }
  }, []);

  const handleResumeFromSavedChange = useCallback((nextValue) => {
    setResumeFromSaved(nextValue);
    if (!nextValue) {
      setLoadCheckpoint('');
    }
  }, []);

  const handleDownloadWeights = useCallback((gameId, checkpointName) => {
    if (!gameId) return;
    const gameKey = gameId.replace(/\//g, '_');
    let url = `/api/models/${gameKey}/download`;

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
      addLog(`Starting training for ${selectedGame}...`);
      analyticsService.trackTrainingStart(selectedGame);
      socketService.emit(config.events.startTraining, {
        game: selectedGame,
        loadCheckpoint: resumeFromSaved ? (loadCheckpoint || null) : null,
        resumeFromSaved
      });
    }
  }, [selectedGame, loadCheckpoint, resumeFromSaved, addLog]);
  
  // Stop training
  const handleStop = useCallback(() => {
    addLog('Stopping training...');
    socketService.emit(config.events.stopTraining);
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

  const handleShare = useCallback(() => {
    const shareData = {
      title: 'Atari RL Training Dashboard',
      text: 'Check out my Atari RL training session.',
      url: window?.location?.href || '',
    };
    if (navigator.share) {
      navigator.share(shareData).catch(() => {});
    } else if (navigator.clipboard && shareData.url) {
      navigator.clipboard.writeText(shareData.url);
    }
  }, []);

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
              <MDBox display="flex" alignItems="center" gap={2}>
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
                    <Tooltip title="Monitor training, metrics, and controls for Rainbow DQN agents.">
                      <Icon fontSize="small" sx={{ color: 'rgba(226,232,240,0.7)' }}>
                        info
                      </Icon>
                    </Tooltip>
                  </MDBox>
                </MDBox>
              </MDBox>

              <MDBox display="flex" alignItems="center" gap={1.5}>
                <MDButton
                  variant="outlined"
                  color="info"
                  size="small"
                  onClick={handleShare}
                  startIcon={<Icon sx={{ fontSize: '1.1rem !important' }}>share</Icon>}
                  sx={{ textTransform: 'none' }}
                >
                  Share
                </MDButton>
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
            </MDBox>
          </Card>
        </MDBox>

        {/* Row 1: Live Game + Controls/Stats */}
        <Grid container spacing={2} mb={2} alignItems="stretch">
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
            <Grid container spacing={2} sx={{ height: '100%' }}>
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
                  setLoadCheckpoint={setLoadCheckpoint}
                  resumeFromSaved={resumeFromSaved}
                  onResumeFromSavedChange={handleResumeFromSavedChange}
                  onDownloadWeights={handleDownloadWeights}
                  checkpointRefreshKey={checkpointRefreshKey}
                  onCheckpointsLoaded={handleCheckpointsLoaded}
                  onDeleteCheckpoint={handleDeleteCheckpoint}
                  trainingSpeed={trainingSpeed}
                  onTrainingSpeedChange={handleTrainingSpeedChange}
                />
              </Grid>
              <Grid item xs={12}>
                <StatsDisplay stats={stats} />
              </Grid>
            </Grid>
          </Grid>
        </Grid>

        {/* Row 2: Metrics + Leaderboard/Challenges */}
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
                    <MDTypography variant="h5" fontWeight="medium">
                      Metrics
                    </MDTypography>
                  </MDBox>
                </MDBox>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <MDBox p={2}>
                        <MDBox display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
                          <MDTypography variant="subtitle2" fontWeight="medium" sx={{ fontSize: '0.95rem' }}>
                            Episode Reward Trend
                          </MDTypography>
                          <Tooltip title="Shows how rewards evolve per episode. Upward trends indicate learning progress.">
                            <Icon sx={{ color: '#0ea5e9', fontSize: '1rem !important' }}>emoji_events</Icon>
                          </Tooltip>
                        </MDBox>
                        {hasEpisodes ? (
                          <RewardChart episodes={episodes} />
                        ) : (
                          <MDBox
                            sx={{
                              border: '1px dashed rgba(148, 163, 184, 0.4)',
                              borderRadius: '10px',
                              p: 2,
                              textAlign: 'center',
                              color: 'text.secondary',
                              fontSize: '0.9rem',
                            }}
                          >
                            Chart placeholder: start training to see reward trends.
                          </MDBox>
                        )}
                      </MDBox>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <MDBox p={2}>
                        <MDBox display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
                          <MDTypography variant="subtitle2" fontWeight="medium" sx={{ fontSize: '0.95rem' }}>
                            Training Loss
                          </MDTypography>
                          <Tooltip title="Tracks prediction error per episode. Lower loss typically means better policy updates.">
                            <Icon sx={{ color: '#ef4444', fontSize: '1rem !important' }}>trending_down</Icon>
                          </Tooltip>
                        </MDBox>
                        {hasEpisodes ? (
                          <LossChart episodes={episodes} />
                        ) : (
                          <MDBox
                            sx={{
                              border: '1px dashed rgba(148, 163, 184, 0.4)',
                              borderRadius: '10px',
                              p: 2,
                              textAlign: 'center',
                              color: 'text.secondary',
                              fontSize: '0.9rem',
                            }}
                          >
                            Chart placeholder: start training to see loss trends.
                          </MDBox>
                        )}
                      </MDBox>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <MDBox p={2}>
                        <MDBox display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
                          <MDTypography variant="subtitle2" fontWeight="medium" sx={{ fontSize: '0.95rem' }}>
                            Q-Values (Mean & Max)
                          </MDTypography>
                          <Tooltip title="Mean and max Q-values show action-value confidence. Healthy growth signals better policies.">
                            <Icon sx={{ color: '#06b6d4', fontSize: '1rem !important' }}>psychology</Icon>
                          </Tooltip>
                        </MDBox>
                        {hasEpisodes ? (
                          <QValueChart episodes={episodes} />
                        ) : (
                          <MDBox
                            sx={{
                              border: '1px dashed rgba(148, 163, 184, 0.4)',
                              borderRadius: '10px',
                              p: 2,
                              textAlign: 'center',
                              color: 'text.secondary',
                              fontSize: '0.9rem',
                            }}
                          >
                            Chart placeholder: start training to see Q-value trends.
                          </MDBox>
                        )}
                      </MDBox>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                      <MDBox p={2}>
                        <MDBox display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
                          <MDTypography variant="subtitle2" fontWeight="medium" sx={{ fontSize: '0.95rem' }}>
                            Action Distribution
                          </MDTypography>
                          <Tooltip title="Shows how often each action is taken. Balanced distributions indicate exploration; skewed ones show exploitation.">
                            <Icon sx={{ color: '#8b5cf6', fontSize: '1rem !important' }}>donut_large</Icon>
                          </Tooltip>
                        </MDBox>
                        {hasActionDist ? (
                          <ActionChart 
                            actionDist={actionDist} 
                            numActions={numActions}
                            actionNames={actionNames}
                          />
                        ) : (
                          <MDBox
                            sx={{
                              border: '1px dashed rgba(148, 163, 184, 0.4)',
                              borderRadius: '10px',
                              p: 2,
                              textAlign: 'center',
                              color: 'text.secondary',
                              fontSize: '0.9rem',
                            }}
                          >
                            Chart placeholder: start training to populate action distribution.
                          </MDBox>
                        )}
                      </MDBox>
                    </Card>
                  </Grid>
                </Grid>
              </MDBox>
            </Card>
          </Grid>

          <Grid item xs={12} lg={4}>
            <Grid container spacing={2} sx={{ height: '100%' }}>
              <Grid item xs={12}>
                <EnhancedLeaderboard currentGame={selectedGame} />
              </Grid>
              <Grid item xs={12}>
                <ChallengesPanel visitorId={localStorage.getItem('visitor_id')} />
              </Grid>
            </Grid>
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
                <Icon color="info">terminal</Icon>
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
                             log.type === 'warning' ? 'warning.main' : 'text.secondary',
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
