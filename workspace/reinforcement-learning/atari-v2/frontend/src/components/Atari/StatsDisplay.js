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

function StatsDisplay({ stats }) {
  const statCards = [
    {
      key: 'episode',
      title: 'Episode',
      value: stats.episode || 0,
      icon: 'sports_esports',
      color: '#f59e0b',
      tooltip: 'How many complete games the AI has played. Each episode runs from start to game over.',
      subtitle: 'Current game'
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

  return (
    <Card
      sx={{
        background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
        border: '1px solid rgba(148, 163, 184, 0.18)',
        boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
        borderRadius: '16px',
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
            <Tooltip 
              key={stat.key}
              title={stat.tooltip}
              arrow
              placement="top"
              enterDelay={200}
            >
              <MDBox
                sx={{
                  padding: '12px',
                  borderRadius: '10px',
                  backgroundColor: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  cursor: 'help',
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
                  <Icon sx={{ color: stat.color, fontSize: '1rem !important' }}>{stat.icon}</Icon>
                </MDBox>
                <MDTypography
                  variant="h6"
                  fontWeight="bold"
                  sx={{ color: stat.color, lineHeight: 1.05, fontSize: '1.1rem' }}
                >
                  {stat.value}
                </MDTypography>
                <MDTypography
                  variant="caption"
                  color="text"
                  sx={{ opacity: 0.6, fontSize: '0.7rem', whiteSpace: 'nowrap' }}
                >
                  {stat.subtitle}
                </MDTypography>
              </MDBox>
            </Tooltip>
          ))}
        </MDBox>
      </MDBox>
    </Card>
  );
}

export default StatsDisplay;
