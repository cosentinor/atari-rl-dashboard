/**
 * Educational Tooltip Component
 * Provides helpful explanations for RL terms
 */

import Tooltip from "@mui/material/Tooltip";
import Icon from "@mui/material/Icon";
import MDBox from "components/MDBox";

const glossary = {
  episode: {
    title: "Episode",
    description: "One complete game from start to game over. The AI plays many episodes to learn."
  },
  reward: {
    title: "Reward",
    description: "The score achieved in the game. Higher rewards mean better performance."
  },
  loss: {
    title: "Training Loss",
    description: "How wrong the AI's predictions are. Lower is better - means AI is learning."
  },
  qvalue: {
    title: "Q-Value",
    description: "AI's confidence about future rewards. Higher means it expects to score more points."
  },
  fps: {
    title: "FPS (Frames Per Second)",
    description: "How fast the game is being rendered. Higher FPS = smoother visualization."
  },
  speed: {
    title: "Training Speed",
    description: "How fast the AI trains. 1x = normal, 2x = twice as fast, 4x = four times faster."
  },
  checkpoint: {
    title: "Checkpoint",
    description: "A saved snapshot of the AI model. You can resume training from any checkpoint."
  },
  action: {
    title: "Action Distribution",
    description: "Which game controls (left, right, fire, etc.) the AI uses most often."
  }
};

function EducationalTooltip({ term, children }) {
  const info = glossary[term];
  
  if (!info) return children;

  return (
    <Tooltip
      title={
        <MDBox p={1}>
          <MDTypography variant="caption" fontWeight="bold" color="white" display="block" mb={0.5}>
            {info.title}
          </MDTypography>
          <MDTypography variant="caption" color="white">
            {info.description}
          </MDTypography>
        </MDBox>
      }
      arrow
      placement="top"
    >
      <MDBox display="inline-flex" alignItems="center" gap={0.5} sx={{ cursor: 'help' }}>
        {children}
        <Icon sx={{ fontSize: '0.875rem', opacity: 0.6 }}>help_outline</Icon>
      </MDBox>
    </Tooltip>
  );
}

export default EducationalTooltip;
