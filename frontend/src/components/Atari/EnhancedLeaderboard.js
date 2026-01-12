/**
 * Enhanced Leaderboard Component
 * Compact version for sidebar - shows top scores with filters
 */

import { useState, useEffect } from 'react';
import Card from "@mui/material/Card";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import CircularProgress from "@mui/material/CircularProgress";
import Icon from "@mui/material/Icon";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import MDButton from "components/MDButton";
import config from "config";

function EnhancedLeaderboard({ currentGame }) {
  const [leaderboardData, setLeaderboardData] = useState([]);
  const [selectedGame, setSelectedGame] = useState('all');
  const [timeframe, setTimeframe] = useState('all_time');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLeaderboard();
    const interval = setInterval(fetchLeaderboard, 30000);
    return () => clearInterval(interval);
  }, [selectedGame, timeframe]);

  const fetchLeaderboard = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (selectedGame !== 'all') params.append('game_id', selectedGame);
      if (timeframe !== 'all_time') params.append('timeframe', timeframe);

      const response = await fetch(`${config.API_BASE_URL}/api/leaderboard?${params}`);
      if (!response.ok) {
        // API not available yet - show empty state
        setLeaderboardData([]);
        return;
      }
      const data = await response.json();
      if (data.success) {
        setLeaderboardData(data.leaderboard || []);
      }
    } catch (err) {
      // Silently fail - API may not be implemented yet
      setLeaderboardData([]);
    } finally {
      setLoading(false);
    }
  };

  const getBadgeIcon = (rank) => {
    if (rank === 1) return { icon: 'military_tech', color: '#facc15' };
    if (rank === 2) return { icon: 'military_tech', color: '#a1a1aa' };
    if (rank === 3) return { icon: 'military_tech', color: '#f97316' };
    return { icon: 'emoji_events', color: 'rgba(226,232,240,0.65)' };
  };

  const formatGameName = (gameId) => {
    return gameId?.split('/')[1]?.replace('-v5', '') || 'Unknown';
  };

  const selectSx = {
    height: '36px',
    fontSize: '0.75rem',
    backgroundColor: 'rgba(255,255,255,0.05)',
    '& .MuiSelect-select': {
      py: 0.5,
      px: 1,
    },
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: 'rgba(255,255,255,0.2)',
    },
  };

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
        <MDBox display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <MDTypography variant="h6" fontWeight="medium" display="flex" alignItems="center" gap={0.75}>
            <Icon sx={{ fontSize: '1.1rem !important', color: '#0ea5e9' }}>leaderboard</Icon>
            Leaderboard
          </MDTypography>
          <MDButton
            variant="text"
            color="info"
            size="small"
            onClick={fetchLeaderboard}
            disabled={loading}
            sx={{ minWidth: 'auto', px: 1 }}
          >
            Refresh
          </MDButton>
        </MDBox>

        {/* Compact Filters - Stacked */}
        <MDBox mb={2}>
          <FormControl fullWidth size="small" sx={{ mb: 1 }}>
            <InputLabel sx={{ fontSize: '0.75rem' }}>Game</InputLabel>
            <Select
              value={selectedGame}
              onChange={(e) => setSelectedGame(e.target.value)}
              label="Game"
              sx={selectSx}
            >
              <MenuItem value="all">All Games</MenuItem>
              <MenuItem value="ALE/Pong-v5">Pong</MenuItem>
              <MenuItem value="ALE/Breakout-v5">Breakout</MenuItem>
              <MenuItem value="ALE/SpaceInvaders-v5">Space Invaders</MenuItem>
              <MenuItem value="ALE/MsPacman-v5">Ms. Pac-Man</MenuItem>
              <MenuItem value="ALE/Asteroids-v5">Asteroids</MenuItem>
              <MenuItem value="ALE/Boxing-v5">Boxing</MenuItem>
              <MenuItem value="ALE/BeamRider-v5">Beam Rider</MenuItem>
              <MenuItem value="ALE/Seaquest-v5">Seaquest</MenuItem>
              <MenuItem value="ALE/Enduro-v5">Enduro</MenuItem>
              <MenuItem value="ALE/Freeway-v5">Freeway</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth size="small">
            <InputLabel sx={{ fontSize: '0.75rem' }}>Period</InputLabel>
            <Select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              label="Period"
              sx={selectSx}
            >
              <MenuItem value="all_time">All Time</MenuItem>
              <MenuItem value="today">Today</MenuItem>
              <MenuItem value="week">This Week</MenuItem>
              <MenuItem value="month">This Month</MenuItem>
            </Select>
          </FormControl>
        </MDBox>

        {/* Leaderboard List */}
        {loading ? (
          <MDBox display="flex" justifyContent="center" py={3}>
            <CircularProgress color="info" size={24} />
          </MDBox>
        ) : leaderboardData.length === 0 ? (
          <MDBox textAlign="center" py={3}>
            <MDTypography variant="caption" color="text">
              No sessions yet. Be the first to train!
            </MDTypography>
          </MDBox>
        ) : (
          <MDBox sx={{ maxHeight: '200px', overflowY: 'auto' }}>
            {leaderboardData.slice(0, 5).map((entry, i) => (
              <MDBox 
                key={entry.session_id}
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                py={0.75}
                px={1}
                mb={0.5}
                sx={{
                  backgroundColor: i < 3 ? 'rgba(6, 182, 212, 0.08)' : 'rgba(255,255,255,0.02)',
                  borderRadius: '6px',
                  borderLeft: i < 3 ? '3px solid #06b6d4' : '3px solid transparent',
                }}
              >
                <MDBox display="flex" alignItems="center" gap={1}>
                  <Icon
                    sx={{
                      fontSize: '1rem !important',
                      color: getBadgeIcon(i + 1).color,
                      minWidth: '18px',
                    }}
                  >
                    {getBadgeIcon(i + 1).icon}
                  </Icon>
                  <MDTypography variant="caption" color="text" sx={{ fontSize: '0.7rem' }}>
                    {formatGameName(entry.game_id)}
                  </MDTypography>
                </MDBox>
                <MDTypography variant="button" fontWeight="bold" color="warning" sx={{ fontSize: '0.75rem' }}>
                  {entry.best_reward?.toFixed(0)}
                </MDTypography>
              </MDBox>
            ))}
          </MDBox>
        )}
      </MDBox>
    </Card>
  );
}

export default EnhancedLeaderboard;
