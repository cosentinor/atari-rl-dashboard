/**
 * Action Distribution Chart - Styled to match requested UI
 */

import Card from "@mui/material/Card";
import Icon from "@mui/material/Icon";
import Tooltip from "@mui/material/Tooltip";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import { Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";

ChartJS.register(ArcElement, ChartTooltip, Legend);

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

const palette = [
  '#0bc5ea', '#22c55e', '#f59e0b', '#a855f7',
  '#3b82f6', '#ef4444', '#14b8a6', '#eab308'
];

function ActionChart({ actionDist, numActions, actionNames }) {
  const labels = actionNames && actionNames.length > 0
    ? actionNames.slice(0, numActions || 6)
    : Array.from({ length: numActions || 6 }, (_, i) => `Action ${i}`);

  const counts = labels.map((_, i) => actionDist[i] || 0);
  const total = counts.reduce((sum, value) => sum + value, 0);

  const data = {
    labels,
    datasets: [
      {
        label: 'Actions Taken',
        data: counts,
        backgroundColor: palette.slice(0, labels.length),
        borderColor: 'rgba(15, 22, 40, 0.8)',
        borderWidth: 2,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#e5e7eb',
          font: { size: 12 },
        },
      },
    },
  };

  return (
    <Card sx={cardSx}>
      <MDBox px={2.5} py={2} display="flex" alignItems="center" justifyContent="space-between">
        <MDTypography variant="h6" fontWeight="medium" color="white">
          Action Distribution
        </MDTypography>
        <Tooltip
          title="Shows which actions the agent uses most often."
          arrow
          componentsProps={{ tooltip: { sx: tooltipSx }, arrow: { sx: { color: tooltipSx.backgroundColor } } }}
        >
          <Icon sx={{ fontSize: '1rem !important', color: '#8b5cf6', cursor: 'help' }}>
            donut_large
          </Icon>
        </Tooltip>
      </MDBox>
      <MDBox px={1.5} pb={1.5} sx={{ height: 260, display: 'flex', alignItems: 'center' }}>
        {total > 0 ? (
          <Doughnut data={data} options={options} />
        ) : (
          <MDTypography variant="body2" color="text" sx={{ opacity: 0.7 }}>
            Action distribution will appear after a few episodes.
          </MDTypography>
        )}
      </MDBox>
    </Card>
  );
}

export default ActionChart;
