/**
 * ModelExplanation Component - Atari RL Training Dashboard
 * Shows details about the selected training level and checkpoint.
 */

import Card from '@mui/material/Card';
import Icon from '@mui/material/Icon';
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";

const LEVEL_COPY = {
  low: {
    label: 'Low',
    title: 'RC_model',
    description: 'Trained by me on rented GPU snapshot between 10k and 50k episodes.',
  },
  medium: {
    label: 'Medium',
    title: 'Bitdefender 10M steps',
    description: 'DQN Modern checkpoint trained for roughly 10M environment steps.',
  },
  high: {
    label: 'High',
    title: 'Bitdefender 50M steps',
    description: 'DQN Modern checkpoint trained for roughly 50M environment steps.',
  },
};

const SOURCE_LABELS = {
  bitdefender: 'Bitdefender',
  sb3: 'SB3 RL Zoo',
  pfrl: 'PFRL Zoo',
  local: "Riccardo's Model",
  rc_model: "Riccardo's Model",
};

const formatNumber = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return value.toLocaleString();
};

const formatAlgorithm = (value) => {
  if (!value) return '-';
  if (value === 'RC_model') return value;
  const normalized = value.replace(/_/g, ' ').toLowerCase();
  if (normalized === 'dqn modern') return 'DQN Modern';
  if (normalized === 'dqn') return 'DQN';
  return normalized.replace(/\b\w/g, (c) => c.toUpperCase());
};

function ModelExplanation({
  selectedGame,
  gameInfo,
  trainingLevel,
  pretrainedModel,
  checkpoint,
  hasPretrainedModels,
}) {
  const levelInfo = LEVEL_COPY[trainingLevel] || LEVEL_COPY.medium;
  const gameLabel = gameInfo?.display_name || gameInfo?.name || selectedGame || 'Select a game';
  const isPretrained = Boolean(pretrainedModel);
  const activeModel = pretrainedModel || checkpoint || null;
  const source = (pretrainedModel?.source || '').toLowerCase();
  const isLocal = source === 'local' || source === 'rc_model';
  const sourceLabel = SOURCE_LABELS[pretrainedModel?.source] || pretrainedModel?.source;
  const headerLabel = `${gameLabel} - ${levelInfo.label}`;

  const descriptionText = isPretrained && !isLocal && source && source !== 'bitdefender'
    ? `Pre-trained checkpoint from ${sourceLabel || 'external source'}.`
    : `${levelInfo.title}. ${levelInfo.description}`;

  const sourceValue = isLocal ? "Riccardo's Model" : (sourceLabel || '-');
  const algorithmValue = isLocal ? 'DQN Rainbow' : formatAlgorithm(activeModel?.algorithm);
  const trainingStepsValue = activeModel?.step ?? activeModel?.episode;
  const seedValue = isLocal ? 0 : activeModel?.seed;


  const cardSx = {
    background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
    borderRadius: '16px',
  };

  const statLabelSx = {
    color: 'rgba(226, 232, 240, 0.6)',
    fontSize: '0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
  };

  const statValueSx = {
    color: '#f8fafc',
    fontWeight: 600,
    fontSize: '0.95rem',
  };

  return (
    <Card sx={cardSx}>
      <MDBox p={{ xs: 2, sm: 2.5 }}>
        <MDBox display="flex" alignItems="center" gap={1} mb={1.5}>
          <Icon sx={{ color: '#0ea5e9' }}>psychology</Icon>
          <MDTypography variant="h6" fontWeight="medium" sx={{ color: '#f8fafc' }}>
            Model Snapshot
          </MDTypography>
        </MDBox>

        <MDTypography variant="button" sx={{ color: 'rgba(226, 232, 240, 0.8)' }}>
          {headerLabel}
        </MDTypography>

        {!selectedGame && (
          <MDTypography variant="body2" sx={{ color: 'rgba(226, 232, 240, 0.7)', mt: 1.5 }}>
            Select a game to see the pre-trained model summary.
          </MDTypography>
        )}

        {selectedGame && !hasPretrainedModels && (
          <MDTypography variant="body2" sx={{ color: 'rgba(226, 232, 240, 0.7)', mt: 1.5 }}>
            No pre-trained models yet. Fetch weights or start a training run to generate snapshots.
          </MDTypography>
        )}

        {selectedGame && hasPretrainedModels && !activeModel && (
          <MDTypography variant="body2" sx={{ color: 'rgba(226, 232, 240, 0.7)', mt: 1.5 }}>
            Model details are unavailable for this level. Try another level or refresh the models.
          </MDTypography>
        )}

        {selectedGame && hasPretrainedModels && activeModel && (
          <>
            <MDTypography variant="body2" sx={{ color: 'rgba(226, 232, 240, 0.8)', mt: 1.5 }}>
              {descriptionText}
            </MDTypography>

            <MDBox
              mt={2}
              display="grid"
              gridTemplateColumns={{ xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' }}
              gap={1.5}
            >
              <MDBox>
                <MDTypography sx={statLabelSx}>Source</MDTypography>
                <MDTypography sx={statValueSx}>{sourceValue}</MDTypography>
              </MDBox>
              <MDBox>
                <MDTypography sx={statLabelSx}>Algorithm</MDTypography>
                <MDTypography sx={statValueSx}>{algorithmValue}</MDTypography>
              </MDBox>
              <MDBox>
                <MDTypography sx={statLabelSx}>Training Steps</MDTypography>
                <MDTypography sx={statValueSx}>{formatNumber(trainingStepsValue)}</MDTypography>
              </MDBox>
              <MDBox>
                <MDTypography sx={statLabelSx}>Seed</MDTypography>
                <MDTypography sx={statValueSx}>{formatNumber(seedValue)}</MDTypography>
              </MDBox>
            </MDBox>

          </>
        )}
      </MDBox>
    </Card>
  );
}

export default ModelExplanation;
