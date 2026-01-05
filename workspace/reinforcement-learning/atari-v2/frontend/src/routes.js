/**
=========================================================
* Atari RL Training Dashboard - Routes
=========================================================
* Simplified routing - Only Atari RL Training Dashboard
*/

// Atari RL Training Dashboard
import AtariDashboard from "layouts/atari";

// @mui icons
import Icon from "@mui/material/Icon";

const routes = [
  {
    type: "collapse",
    name: "Atari RL Training",
    key: "atari-dashboard",
    icon: <Icon fontSize="small">sports_esports</Icon>,
    route: "/atari",
    component: <AtariDashboard />,
  },
];

export default routes;
