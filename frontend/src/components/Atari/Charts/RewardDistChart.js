/**
 * Reward Distribution Chart - Score Distribution Histogram
 */

import VerticalBarChart from "examples/Charts/BarCharts/VerticalBarChart";

function RewardDistChart({ rewardDist }) {
  const chartData = {
    labels: (rewardDist?.bins || []).map(b => b.toFixed(0)),
    datasets: [
      {
        label: "Frequency",
        color: "dark",
        data: rewardDist?.counts || [],
      },
    ],
  };

  return (
    <VerticalBarChart
      icon={{ color: "dark", component: "bar_chart" }}
      title="Score Distribution"
      description="Range and consistency of scores"
      height="18rem"
      chart={chartData}
    />
  );
}

export default RewardDistChart;
