/**
 * ControlPanel Component - Atari RL Training Dashboard
 * Compact game selection, speed controls, and training buttons
 */

import { useState, useEffect } from 'react';
import Card from '@mui/material/Card';
import Icon from '@mui/material/Icon';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import { useTheme } from '@mui/material/styles';
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import MDButton from "components/MDButton";
import config from "config";

function ControlPanel({
  games,
  selectedGame,
  onGameChange,
  isTraining,
  onStart,
  onStop,
  onSave,
  loadCheckpoint,
  resumeFromSaved,
  onLoadCheckpointChange,
  onResumeFromSavedChange,
  onDownloadWeights,
  pretrainedModel,
  pretrainedLoading,
  checkpointRefreshKey,
  onCheckpointsLoaded,
  trainingSpeed,
  onTrainingSpeedChange,
  trainingLevel,
  onTrainingLevelChange,
  hasPretrainedModels,
}) {
  const theme = useTheme();
  const [checkpoints, setCheckpoints] = useState([]);
  const isPretrainedSelected = Boolean(pretrainedModel);
  const fontFamily = theme.typography?.fontFamily || '"Inter", "Helvetica", "Arial", sans-serif';

  // Fetch checkpoints when game changes
  useEffect(() => {
    if (selectedGame) {
      const gameKey = selectedGame.replace(/\//g, '_');
      fetch(`${config.API_BASE_URL}/api/models/${gameKey}`)
        .then(r => r.json())
        .then(data => {
          if (data.success) {
            const next = data.checkpoints || [];
            setCheckpoints(next);
            if (onCheckpointsLoaded) onCheckpointsLoaded(next);
          }
        })
        .catch(console.error);
    } else {
      setCheckpoints([]);
      if (onCheckpointsLoaded) onCheckpointsLoaded([]);
    }
  }, [selectedGame, checkpointRefreshKey, onCheckpointsLoaded]);

  const speedOptions = [
    { value: '1x', label: '1x' },
    { value: '2x', label: '2x' },
    { value: '4x', label: '4x' },
  ];

  const trainingLevelOptions = [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
  ];

  const canDownloadWeights = Boolean(pretrainedModel?.id) || checkpoints.length > 0;

  const cardSx = {
    background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
    borderRadius: '16px',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
    overflow: 'hidden',
    position: 'relative',
  };

  const sectionLabelSx = {
    color: 'rgba(226, 232, 240, 0.72)',
    fontSize: '0.85rem',
    fontWeight: 500,
    mb: 0.75,
  };

  const selectSx = {
    height: '46px',
    backgroundColor: 'rgba(15, 23, 42, 0.6)',
    borderRadius: '12px',
    fontFamily,
    color: '#e2e8f0',
    '& .MuiSelect-select': {
      display: 'flex',
      alignItems: 'center',
      padding: '10px 12px',
      fontFamily,
    },
    '& .MuiSelect-select.Mui-disabled': {
      WebkitTextFillColor: 'rgba(226, 232, 240, 0.6)',
    },
    '& .MuiInputBase-input.Mui-disabled': {
      WebkitTextFillColor: 'rgba(226, 232, 240, 0.6)',
      color: 'rgba(226, 232, 240, 0.6)',
    },
    '& .MuiSelect-icon': {
      color: 'rgba(226, 232, 240, 0.7)',
    },
    '& .MuiSelect-icon.Mui-disabled': {
      color: 'rgba(226, 232, 240, 0.5)',
    },
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: 'rgba(148, 163, 184, 0.2)',
    },
    '&:hover .MuiOutlinedInput-notchedOutline': {
      borderColor: 'rgba(56, 189, 248, 0.6)',
    },
    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
      borderColor: 'rgba(56, 189, 248, 0.8)',
      boxShadow: '0 0 0 2px rgba(14, 165, 233, 0.2)',
    },
    '&.Mui-disabled .MuiOutlinedInput-notchedOutline': {
      borderColor: 'rgba(148, 163, 184, 0.25)',
    },
  };

  const menuProps = {
    PaperProps: {
      sx: {
        mt: 1,
        backgroundColor: '#0f172a',
        border: '1px solid rgba(148, 163, 184, 0.15)',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)',
        '& .MuiMenuItem-root': {
          color: '#e2e8f0',
          fontSize: '0.95rem',
          fontFamily,
          '&:hover': {
            backgroundColor: 'rgba(148, 163, 184, 0.12)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(56, 189, 248, 0.2)',
          },
          '&.Mui-selected:hover': {
            backgroundColor: 'rgba(56, 189, 248, 0.28)',
          },
        },
      },
    },
  };

  const speedGroupSx = {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    borderRadius: '12px',
    border: '1px solid rgba(139, 92, 246, 0.55)',
    backgroundColor: 'rgba(15, 23, 42, 0.6)',
    overflow: 'hidden',
  };

  const speedButtonSx = (isActive, index, total) => ({
    borderRadius: 0,
    minHeight: '42px',
    fontSize: '0.9rem',
    fontWeight: 600,
    textTransform: 'none',
    color: isActive ? '#f8fafc' : '#8b5cf6',
    background: isActive
      ? 'linear-gradient(90deg, #06b6d4 0%, #0ea5e9 100%)'
      : 'transparent',
    borderRight: index < total - 1 ? '1px solid rgba(139, 92, 246, 0.35)' : 'none',
    '&:hover': {
      background: isActive
        ? 'linear-gradient(90deg, #06b6d4 0%, #0ea5e9 100%)'
        : 'rgba(139, 92, 246, 0.14)',
    },
  });

  const levelGroupSx = {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    borderRadius: '12px',
    border: '1px solid rgba(245, 158, 11, 0.55)',
    backgroundColor: 'rgba(15, 23, 42, 0.6)',
    overflow: 'hidden',
  };

  const levelButtonSx = (isActive, index, total) => ({
    borderRadius: 0,
    minHeight: '42px',
    fontSize: '0.9rem',
    fontWeight: 600,
    textTransform: 'none',
    color: '#f8fafc',
    background: 'linear-gradient(90deg, #f59e0b 0%, #f97316 100%)',
    borderRight: index < total - 1 ? '1px solid rgba(245, 158, 11, 0.35)' : 'none',
    '&:hover': {
      background: 'linear-gradient(90deg, #f59e0b 0%, #f97316 100%)',
    },
  });


  const actionGridSx = {
    display: 'grid',
    gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, 1fr)' },
    gap: 2,
  };

  const actionButtonBaseSx = {
    minHeight: '76px',
    borderRadius: '14px',
    textTransform: 'none',
    fontSize: '0.9rem',
    fontWeight: 600,
    lineHeight: 1.1,
    padding: '12px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 1,
    boxShadow: '0 14px 30px rgba(0, 0, 0, 0.35)',
    '& .MuiIcon-root': {
      fontSize: '1.4rem',
    },
    '&.Mui-disabled': {
      background: 'rgba(15, 23, 42, 0.6)',
      color: 'rgba(226, 232, 240, 0.4)',
      boxShadow: 'none',
    },
  };

  const selectedGameInfo = selectedGame
    ? games.find((game) => game.id === selectedGame)
    : null;
  const trainedEpisodes = selectedGameInfo?.trained_episodes;
  const formatEpisodeCount = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return null;
    return value.toLocaleString();
  };

  const sortedCheckpoints = [...checkpoints].sort(
    (a, b) => (b.episode ?? 0) - (a.episode ?? 0)
  );
  const latestCheckpoint = sortedCheckpoints[0] || null;
  const latestEpisodeLabel = formatEpisodeCount(latestCheckpoint?.episode);
  const maxTrainingEpisodes =
    typeof trainedEpisodes === 'number' && trainedEpisodes > 0 ? trainedEpisodes : null;
  const maxTrainingLabel = formatEpisodeCount(maxTrainingEpisodes);
  const halfTrainingEpisodes = maxTrainingEpisodes
    ? Math.round(maxTrainingEpisodes / 2)
    : null;
  const halfTrainingLabel = formatEpisodeCount(halfTrainingEpisodes);

  const findClosestCheckpoint = (targetEpisode) => {
    if (
      targetEpisode === null ||
      targetEpisode === undefined ||
      Number.isNaN(targetEpisode) ||
      sortedCheckpoints.length === 0
    ) {
      return null;
    }
    return sortedCheckpoints.reduce((closest, checkpoint) => {
      if (!closest) return checkpoint;
      const currentDistance = Math.abs((checkpoint.episode ?? 0) - targetEpisode);
      const bestDistance = Math.abs((closest.episode ?? 0) - targetEpisode);
      return currentDistance < bestDistance ? checkpoint : closest;
    }, null);
  };

  const halfCheckpoint = halfTrainingEpisodes
    ? findClosestCheckpoint(halfTrainingEpisodes)
    : null;
  const maxCheckpoint = maxTrainingEpisodes
    ? findClosestCheckpoint(maxTrainingEpisodes)
    : latestCheckpoint;

  const halfOptionLabel = halfTrainingLabel
    ? `Halfway (Ep ${halfTrainingLabel})`
    : 'Halfway';

  let maxOptionLabel = maxTrainingLabel
    ? `Max training (Ep ${maxTrainingLabel})`
    : 'Last saved';
  if (!maxTrainingLabel && latestEpisodeLabel) {
    maxOptionLabel = `Last saved (Ep ${latestEpisodeLabel})`;
  }
  if (
    maxTrainingLabel &&
    latestEpisodeLabel &&
    latestCheckpoint?.episode !== maxTrainingEpisodes
  ) {
    maxOptionLabel = `${maxOptionLabel} - last saved Ep ${latestEpisodeLabel}`;
  }

  const startFromValue = !resumeFromSaved
    ? 'fresh'
    : halfCheckpoint && loadCheckpoint === halfCheckpoint.filename
    ? 'half'
    : 'max';

  const handleStartFromChange = (event) => {
    const nextValue = event.target.value;
    if (nextValue === 'fresh') {
      onResumeFromSavedChange?.(false);
      onLoadCheckpointChange?.('');
      return;
    }

    onResumeFromSavedChange?.(true);
    if (nextValue === 'half') {
      onLoadCheckpointChange?.(halfCheckpoint?.filename || '');
      return;
    }

    onLoadCheckpointChange?.(maxCheckpoint?.filename || '');
  };

  const showStartFrom = selectedGame && !isPretrainedSelected;

  const disableLevelControls = pretrainedLoading;
  const startLabel = isPretrainedSelected ? 'Play' : (resumeFromSaved ? 'Resume' : 'Start');

  return (
    <Card sx={cardSx}>
      <MDBox p={{ xs: 2, sm: 2.5 }} sx={{ position: 'relative', zIndex: 1 }}>
        <MDTypography variant="h6" fontWeight="medium" sx={{ color: '#f8fafc', mb: 1.5 }} display="flex" alignItems="center" gap={0.75}>
          <Icon sx={{ fontSize: '1.3rem !important', color: '#0ea5e9' }}>tune</Icon>
          Training Controls
        </MDTypography>

        <MDBox display="flex" flexDirection="column" gap={2}>
          <MDBox>
            <MDTypography sx={sectionLabelSx}>Select Game</MDTypography>
            <Select
              value={selectedGame}
              onChange={(e) => onGameChange(e.target.value)}
              disabled={isTraining}
              displayEmpty
              fullWidth
              sx={selectSx}
              MenuProps={menuProps}
              renderValue={(selected) => {
                if (!selected) {
                  return <span style={{ color: 'rgba(226, 232, 240, 0.6)', fontFamily }}>Select Game</span>;
                }
                const game = games.find((g) => g.id === selected);
                return game?.display_name || game?.name || selected;
              }}
            >
              {games.map((game) => (
                <MenuItem key={game.id} value={game.id}>
                  {game.display_name || game.name}
                </MenuItem>
              ))}
            </Select>
          </MDBox>

          {selectedGame && canDownloadWeights && (
            <MDBox>
              <MDButton
                variant="outlined"
                size="small"
                onClick={() => onDownloadWeights?.(selectedGame, loadCheckpoint, pretrainedModel?.id)}
                disabled={!onDownloadWeights}
                sx={{
                  borderColor: 'rgba(56, 189, 248, 0.6)',
                  color: '#e2e8f0',
                  textTransform: 'none',
                  '&:hover': {
                    borderColor: 'rgba(56, 189, 248, 0.9)',
                    backgroundColor: 'rgba(14, 165, 233, 0.12)',
                  },
                }}
              >
                <Icon sx={{ mr: 0.5 }}>download</Icon>
                Download Weights
              </MDButton>
            </MDBox>
          )}

          {showStartFrom && (
            <MDBox>
              <MDTypography sx={sectionLabelSx}>Start game from:</MDTypography>
              <Select
                value={startFromValue}
                onChange={handleStartFromChange}
                disabled={isTraining}
                displayEmpty
                fullWidth
                sx={selectSx}
                MenuProps={menuProps}
              >
                <MenuItem value='fresh'>0</MenuItem>
                <MenuItem value='half' disabled={!halfCheckpoint}>
                  {halfCheckpoint ? halfOptionLabel : `${halfOptionLabel} (unavailable)`}
                </MenuItem>
                <MenuItem value='max' disabled={!maxCheckpoint}>
                  {maxOptionLabel}
                </MenuItem>
              </Select>
            </MDBox>
          )}

          <MDBox>

            <MDTypography sx={sectionLabelSx}>Training Level</MDTypography>
            <MDBox sx={levelGroupSx}>
              {trainingLevelOptions.map((option, index) => (
                <MDButton
                  key={option.value}
                  variant="contained"
                  disableElevation
                  disabled={disableLevelControls}
                  onClick={() => onTrainingLevelChange?.(option.value)}
                  sx={levelButtonSx(trainingLevel === option.value, index, trainingLevelOptions.length)}
                >
                  {option.label}
                </MDButton>
              ))}
            </MDBox>
            {selectedGame && !hasPretrainedModels && (
              <MDTypography
                variant="caption"
                sx={{ color: 'rgba(226, 232, 240, 0.6)', display: 'block', mt: 1 }}
              >
                No pre-trained models yet. Fetch weights or train to create snapshots.
              </MDTypography>
            )}
          </MDBox>

          <MDBox>
            <MDTypography sx={sectionLabelSx}>Training Speed</MDTypography>
            <MDBox sx={speedGroupSx}>
              {speedOptions.map((option, index) => (
                <MDButton
                  key={option.value}
                  variant="contained"
                  disableElevation
                  onClick={() => onTrainingSpeedChange(option.value)}
                  sx={speedButtonSx(trainingSpeed === option.value, index, speedOptions.length)}
                >
                  {option.label}
                </MDButton>
              ))}
            </MDBox>
          </MDBox>

          <MDBox sx={actionGridSx}>
            <MDButton
              variant="contained"
              onClick={onStart}
              disabled={!selectedGame || isTraining}
              sx={{
                ...actionButtonBaseSx,
                background: 'linear-gradient(180deg, #22c55e 0%, #15803d 100%)',
                color: '#f8fafc',
                '&:hover': {
                  background: 'linear-gradient(180deg, #16a34a 0%, #166534 100%)',
                },
              }}
            >
              <Icon>play_arrow</Icon>
              {startLabel}
            </MDButton>

            <MDButton
              variant="contained"
              onClick={onStop}
              disabled={!isTraining}
              sx={{
                ...actionButtonBaseSx,
                background: 'linear-gradient(180deg, #f87171 0%, #dc2626 100%)',
                color: '#f8fafc',
                '&:hover': {
                  background: 'linear-gradient(180deg, #ef4444 0%, #b91c1c 100%)',
                },
              }}
            >
              <Icon>stop</Icon>
              Stop
            </MDButton>

            <MDButton
              variant="contained"
              onClick={onSave}
              disabled={!isTraining || isPretrainedSelected}
              sx={{
                ...actionButtonBaseSx,
                background: 'linear-gradient(180deg, #1f2937 0%, #111827 100%)',
                color: 'rgba(226, 232, 240, 0.85)',
                border: '1px solid rgba(148, 163, 184, 0.15)',
                '&:hover': {
                  background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
                },
              }}
            >
              <Icon>save</Icon>
              Save
            </MDButton>
          </MDBox>
        </MDBox>
      </MDBox>
    </Card>
  );
}

export default ControlPanel;