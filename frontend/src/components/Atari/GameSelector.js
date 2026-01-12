/**
 * Visual Game Selector Component
 * Grid of game cards for better UX
 */

import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActionArea from "@mui/material/CardActionArea";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import MDBadge from "components/MDBadge";

const gameEmojis = {
  'Pong': '🏓',
  'Breakout': '🧱',
  'SpaceInvaders': '👾',
  'Asteroids': '🚀',
  'MsPacman': '🎮',
  'Boxing': '🥊',
  'Seaquest': '🌊',
  'BeamRider': '✨',
  'Enduro': '🏎️',
  'Freeway': '🐔'
};

function GameSelector({ games, selectedGame, onGameChange, disabled }) {
  const getGameEmoji = (gameName) => {
    return gameEmojis[gameName] || '🎮';
  };

  const getGameDescription = (game) => {
    return game.description || 'Classic Atari game';
  };

  return (
    <MDBox>
      <MDTypography variant="button" fontWeight="medium" color="text" mb={2} display="block">
        Select Game
      </MDTypography>
      
      <Grid container spacing={2}>
        {games.map((game) => {
          const isSelected = selectedGame === game.id;
          const emoji = getGameEmoji(game.name);
          
          return (
            <Grid item xs={6} sm={4} md={6} lg={4} key={game.id}>
              <Card
                sx={{
                  border: isSelected ? '2px solid' : '1px solid',
                  borderColor: isSelected ? 'info.main' : 'divider',
                  backgroundColor: isSelected ? 'rgba(6, 182, 212, 0.08)' : 'transparent',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: 'info.main',
                    transform: 'translateY(-2px)',
                    boxShadow: 3
                  }
                }}
              >
                <CardActionArea
                  onClick={() => !disabled && onGameChange(game.id)}
                  disabled={disabled}
                >
                  <CardContent sx={{ textAlign: 'center', py: 2, px: 1 }}>
                    <MDTypography variant="h3" mb={1}>
                      {emoji}
                    </MDTypography>
                    <MDTypography 
                      variant="button" 
                      fontWeight="medium"
                      color={isSelected ? 'info' : 'text'}
                      sx={{ fontSize: '0.75rem' }}
                    >
                      {game.display_name || game.name}
                    </MDTypography>
                    {isSelected && (
                      <MDBox mt={1}>
                        <MDBadge 
                          badgeContent="Selected" 
                          color="info" 
                          size="xs"
                          container
                        />
                      </MDBox>
                    )}
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          );
        })}
      </Grid>
      
      {games.length === 0 && (
        <MDBox textAlign="center" py={3}>
          <MDTypography variant="caption" color="text">
            No games available
          </MDTypography>
        </MDBox>
      )}
    </MDBox>
  );
}

export default GameSelector;
