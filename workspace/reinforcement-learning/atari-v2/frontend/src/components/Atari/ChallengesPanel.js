/**
 * Challenges Panel Component
 * Compact sidebar version - shows daily/weekly challenges
 */

import { useState, useEffect } from 'react';
import Card from "@mui/material/Card";
import LinearProgress from "@mui/material/LinearProgress";
import CircularProgress from "@mui/material/CircularProgress";
import Icon from "@mui/material/Icon";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";

function ChallengesPanel({ visitorId }) {
  const [challenges, setChallenges] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchChallenges();
    const interval = setInterval(fetchChallenges, 60000);
    return () => clearInterval(interval);
  }, [visitorId]);

  const fetchChallenges = async () => {
    try {
      const response = await fetch(`/api/challenges?visitor_id=${visitorId || ''}`);
      if (!response.ok) {
        // API not available yet - show empty state
        setChallenges([]);
        return;
      }
      const data = await response.json();
      if (data.success) {
        setChallenges(data.challenges || []);
      }
    } catch (err) {
      // Silently fail - API may not be implemented yet
      setChallenges([]);
    } finally {
      setLoading(false);
    }
  };

  const getChallengeIcon = (type) => {
    switch (type) {
      case 'daily': return 'calendar_today';
      case 'weekly': return 'bar_chart';
      case 'score': return 'my_location';
      case 'episodes': return 'autorenew';
      default: return 'emoji_events';
    }
  };

  const getProgressPercentage = (progress, target) => {
    return Math.min(100, (progress / target) * 100);
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
        <MDTypography variant="h6" fontWeight="medium" mb={2} display="flex" alignItems="center" gap={0.75}>
          <Icon sx={{ fontSize: '1.1rem !important', color: '#0ea5e9' }}>flag</Icon>
          Challenges
        </MDTypography>

        {loading ? (
          <MDBox display="flex" justifyContent="center" py={2}>
            <CircularProgress color="info" size={20} />
          </MDBox>
        ) : challenges.length === 0 ? (
          <MDBox textAlign="center" py={2}>
            <MDTypography variant="caption" color="text">
              No active challenges
            </MDTypography>
            <MDTypography variant="caption" color="text" display="block" mt={0.5} sx={{ opacity: 0.6 }}>
              Start training to unlock challenges!
            </MDTypography>
          </MDBox>
        ) : (
          <MDBox display="flex" flexDirection="column" gap={1.5}>
            {challenges.slice(0, 3).map((challenge, index) => {
              const progress = getProgressPercentage(challenge.progress, challenge.target);
              const isCompleted = challenge.completed;

              return (
                <MDBox
                  key={index}
                  sx={{
                    p: 1.5,
                    borderRadius: '8px',
                    backgroundColor: isCompleted ? 'rgba(76, 175, 80, 0.1)' : 'rgba(255,255,255,0.03)',
                    border: '1px solid',
                    borderColor: isCompleted ? 'rgba(76, 175, 80, 0.3)' : 'rgba(255,255,255,0.05)',
                  }}
                >
                  <MDBox display="flex" alignItems="center" gap={1} mb={1}>
                    <Icon sx={{ fontSize: '1rem !important', color: 'rgba(226,232,240,0.85)' }}>
                      {getChallengeIcon(challenge.type)}
                    </Icon>
                    <MDTypography variant="caption" fontWeight="medium" sx={{ flex: 1, lineHeight: 1.2 }}>
                      {challenge.title}
                    </MDTypography>
                    {isCompleted && (
                      <span style={{ color: '#22c55e', fontSize: '0.9rem' }}>✓</span>
                    )}
                  </MDBox>

                  <LinearProgress
                    variant="determinate"
                    value={progress}
                    color={isCompleted ? 'success' : 'info'}
                    sx={{ height: 4, borderRadius: 2, mb: 0.5 }}
                  />

                  <MDTypography variant="caption" color="text" sx={{ fontSize: '0.65rem', opacity: 0.7 }}>
                    {challenge.progress} / {challenge.target} ({progress.toFixed(0)}%)
                  </MDTypography>
                </MDBox>
              );
            })}
          </MDBox>
        )}
      </MDBox>
    </Card>
  );
}

export default ChallengesPanel;
