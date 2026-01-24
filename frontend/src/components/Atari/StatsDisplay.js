/**
 * StatsDisplay Component - Atari RL Training Dashboard
 * Displays real-time training statistics with educational tooltips
 * Fixed grid layout - uses flexbox for reliable alignment
 */

import Tooltip from "@mui/material/Tooltip";
import Card from "@mui/material/Card";
import Icon from "@mui/material/Icon";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";


const formatNumber = (value) => {
  if (value === null || value === undefined) return '0';
  let numeric = value;
  if (typeof numeric === 'string' && numeric.trim() !== '') {
    const parsed = Number(numeric);
    if (Number.isFinite(parsed)) numeric = parsed;
  }
  if (typeof numeric !== 'number' || Number.isNaN(numeric)) return String(value);
  return Math.floor(numeric).toLocaleString('en-US');
};

const getValueFontSize = (text) => {
  const length = String(text).length;
  if (length <= 7) return '1.1rem';
  if (length <= 9) return '1rem';
  if (length <= 11) return '0.92rem';
  return '0.85rem';
};

function StatsDisplay({ stats }) {
  const stepsValue = formatNumber(stats.totalSteps || 0);
  const episodeValue = formatNumber(stats.episode || 0);
  const stepsTooltip = `Total environment steps completed. Episodes: ${episodeValue}. This matches the checkpoint step count shown in the model snapshot.`;
  const statCards = [
    {
      key: 'steps',
      title: 'Training Steps',
      value: stepsValue,
      valueSx: {
        whiteSpace: 'nowrap',
        fontVariantNumeric: 'tabular-nums',
        letterSpacing: '-0.02em',
        fontSize: getValueFontSize(stepsValue),
      },
      icon: 'sports_esports',
      color: '#f59e0b',
      tooltip: stepsTooltip,
      subtitle: null
    },
    {
      key: 'reward',
      title: 'Reward',
      value: stats.reward || 0,
      icon: 'emoji_events',
      color: '#8b5cf6',
      tooltip: 'The score from the most recent game. This fluctuates as the AI explores different strategies.',
      subtitle: 'Latest score'
    },
    {
      key: 'best',
      title: 'Best',
      value: stats.bestReward || 0,
      icon: 'star',
      color: '#22c55e',
      tooltip: 'The highest score achieved so far. This is the benchmark the AI tries to beat.',
      subtitle: 'High score'
    },
    {
      key: 'loss',
      title: 'Loss',
      value: stats.loss || '0.00',
      icon: 'trending_down',
      color: '#ef4444',
      tooltip: 'Measures prediction error. Lower values mean the AI is learning better.',
      subtitle: 'Lower is better'
    },
    {
      key: 'qvalue',
      title: 'Q-Value',
      value: stats.qValue || '0.00',
      icon: 'psychology',
      color: '#06b6d4',
      tooltip: "The AI's estimate of future rewards. Higher values indicate confidence.",
      subtitle: 'Expected return'
    },
    {
      key: 'fps',
      title: 'FPS',
      value: stats.fps || 0,
      icon: 'speed',
      color: '#3b82f6',
      tooltip: 'Frames rendered per second. Shows visualization speed.',
      subtitle: 'Frame rate'
    }
  ];

  const tooltipSx = {
    backgroundColor: 'rgba(15, 23, 42, 0.82)',
    border: '1px solid rgba(148, 163, 184, 0.3)',
    color: '#e2e8f0',
    fontSize: '0.75rem',
    boxShadow: '0 12px 24px rgba(0,0,0,0.35)',
  };

  return (
    <Card
      sx={{
        background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
        border: '1px solid rgba(148, 163, 184, 0.18)',
        boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
        borderRadius: '16px',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <MDBox p={2}>
        <MDBox
          sx={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
            gap: 1.5,
          }}
        >
          {statCards.map((stat) => (
            <MDBox
              key={stat.key}
              sx={{
                padding: '12px',
                borderRadius: '10px',
                backgroundColor: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                transition: 'all 0.2s ease',
                minHeight: '82px',
              }}
            >
              <MDBox 
                display="flex" 
                alignItems="center" 
                justifyContent="space-between" 
                mb={1}
              >
                <MDTypography
                  sx={{ 
                    color: 'rgba(255,255,255,0.7)', 
                    fontSize: '0.68rem', 
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px'
                  }}
                >
                  {stat.title}
                </MDTypography>
                <Tooltip
                  title={stat.tooltip}
                  arrow
                  placement="top"
                  enterDelay={150}
                  componentsProps={{
                    tooltip: { sx: tooltipSx },
                    arrow: { sx: { color: tooltipSx.backgroundColor } },
                  }}
                >
                  <Icon sx={{ color: stat.color, fontSize: '1rem !important', cursor: 'help' }}>
                    {stat.icon}
                  </Icon>
                </Tooltip>
              </MDBox>
              <MDTypography
                variant="h6"
                fontWeight="bold"
                sx={{
                  color: stat.color,
                  lineHeight: 1.05,
                  fontSize: '1.1rem',
                  ...(stat.valueSx || {}),
                }}
              >
                {stat.value}
              </MDTypography>
              {stat.subtitle && (
                <MDTypography
                  variant="caption"
                  color="text"
                  sx={{ opacity: 0.6, fontSize: '0.7rem', whiteSpace: 'nowrap' }}
                >
                  {stat.subtitle}
                </MDTypography>
              )}
            </MDBox>
          ))}
        </MDBox>
      </MDBox>
    </Card>
  );
}

export default StatsDisplay;
