/**
 * Episode Rewards Chart - Styled to match requested UI
 */

import Card from "@mui/material/Card";
import Icon from "@mui/material/Icon";
import Tooltip from "@mui/material/Tooltip";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, PointElement, CategoryScale, LinearScale, ChartTooltip, Legend);

const cardSx = {
  background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
  border: '1px solid rgba(148, 163, 184, 0.18)',
  boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
  borderRadius: '16px',
};

const tooltipSx = {
  backgroundColor: 'rgba(15, 23, 42, 0.82)',
  border: '1px solid rgba(148, 163, 184, 0.3)',
  color: '#e2e8f0',
  fontSize: '0.75rem',
  boxShadow: '0 12px 24px rgba(0,0,0,0.35)',
};

function RewardChart({ episodes }) {
  const maxPoints = 25;
  const trimmed = episodes.slice(-maxPoints);
  const labels = Array.from({ length: maxPoints }, (_, i) => i + 1);
  const padding = Array(maxPoints - trimmed.length).fill(null);
  const dataValues = trimmed.map(e => e.reward || 0).concat(padding);

  const data = {
    labels,
    datasets: [
      {
        label: "Episode Reward",
        data: dataValues,
        borderColor: '#0bc5ea',
        backgroundColor: 'rgba(11, 197, 234, 0.2)',
        borderWidth: 3,
        pointRadius: 2,
        tension: 0.25,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        labels: {
          color: '#e5e7eb',
          font: { size: 14 },
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        ticks: {
          color: '#9ca3af',
          font: { size: 12 },
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.25)',
        },
      },
      y: {
        ticks: {
          color: '#9ca3af',
          font: { size: 12 },
        },
        grid: {
          color: 'rgba(148, 163, 184, 0.25)',
        },
      },
    },
  };

  return (
    <Card sx={cardSx}>
      <MDBox px={2.5} py={2} display="flex" alignItems="center" justifyContent="space-between">
        <MDTypography variant="h6" fontWeight="medium" color="white">
          Episode Rewards Chart
        </MDTypography>
        <Tooltip
          title="Rewards earned each episode. Higher is better."
          arrow
          componentsProps={{ tooltip: { sx: tooltipSx }, arrow: { sx: { color: tooltipSx.backgroundColor } } }}
        >
          <Icon sx={{ fontSize: '1rem !important', color: '#0bc5ea', cursor: 'help' }}>
            emoji_events
          </Icon>
        </Tooltip>
      </MDBox>
      <MDBox px={1.5} pb={1.5} sx={{ height: 260 }}>
        <Line data={data} options={options} />
      </MDBox>
    </Card>
  );
}

export default RewardChart;
