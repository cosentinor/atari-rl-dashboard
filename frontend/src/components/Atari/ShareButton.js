/**
 * Share Button Component
 * Social sharing for training sessions
 */

import { useState } from 'react';
import { useTheme } from "@mui/material/styles";
import IconButton from "@mui/material/IconButton";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import Icon from "@mui/material/Icon";
import MDButton from "components/MDButton";

function ShareButton({ sessionId, gameId, bestReward, episodes }) {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  
  const open = Boolean(anchorEl);
  const fontFamily = theme.typography?.fontFamily || '"Inter", "Helvetica", "Arial", sans-serif';
  const primaryFont = fontFamily.split(',')[0]?.replace(/['"]/g, '').trim() || 'Inter';

  const shareUrl = `${window.location.origin}/share/${sessionId}`;
  const shareText = `I trained an AI to play ${gameId?.split('/')[1]?.replace('-v5', '') || 'Atari'}! Best score: ${bestReward?.toFixed(0)} in ${episodes} episodes. Watch it play:`;

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleShare = (platform) => {
    let url;
    switch (platform) {
      case 'twitter':
        url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`;
        break;
      case 'facebook':
        url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
        break;
      case 'linkedin':
        url = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
        break;
      case 'reddit':
        url = `https://reddit.com/submit?url=${encodeURIComponent(shareUrl)}&title=${encodeURIComponent(shareText)}`;
        break;
    }

    if (url) {
      window.open(url, '_blank', 'width=600,height=400');
    }

    handleClose();
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setSnackbar({ open: true, message: 'Link copied to clipboard!', severity: 'success' });
    } catch (err) {
      setSnackbar({ open: true, message: 'Failed to copy link', severity: 'error' });
    }
    handleClose();
  };

  const handleDownloadCard = async () => {
    // Generate shareable image card
    if (document.fonts?.load) {
      try {
        await document.fonts.load(`600 60px ${primaryFont}`);
      } catch (err) {
        console.warn('Share card font load failed:', err);
      }
    }

    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 630;
    const ctx = canvas.getContext('2d');

    // Background gradient
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#06b6d4');
    gradient.addColorStop(1, '#8b5cf6');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Title
    ctx.fillStyle = '#ffffff';
    ctx.font = `700 60px ${fontFamily}`;
    ctx.fillText('Atari RL Training', 60, 100);

    // Game name
    const gameName = gameId?.split('/')[1]?.replace('-v5', '') || 'Atari Game';
    ctx.font = `700 48px ${fontFamily}`;
    ctx.fillText(gameName, 60, 200);

    // Stats
    ctx.font = `500 36px ${fontFamily}`;
    ctx.fillText(`Best Score: ${bestReward?.toFixed(0)}`, 60, 300);
    ctx.fillText(`Episodes: ${episodes}`, 60, 360);

    // Footer
    ctx.font = `500 24px ${fontFamily}`;
    ctx.fillText('Watch AI learn to play in real-time', 60, 560);

    // Convert to blob and download
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `atari-rl-${gameName}-${sessionId}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });

    setSnackbar({ open: true, message: 'Card downloaded!', severity: 'success' });
    handleClose();
  };

  if (!sessionId) return null;

  return (
    <>
      <MDButton
        variant="gradient"
        color="info"
        size="small"
        onClick={handleClick}
        iconOnly={false}
      >
        <Icon sx={{ mr: 1 }}>share</Icon>
        Share
      </MDButton>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        PaperProps={{
          sx: {
            mt: 1,
            backgroundColor: '#0f172a',
            border: '1px solid rgba(148, 163, 184, 0.18)',
            boxShadow: '0 18px 40px rgba(0, 0, 0, 0.55)',
            '& .MuiMenuItem-root': {
              color: '#e2e8f0',
              fontFamily,
              fontSize: '0.95rem',
              '&:hover': {
                backgroundColor: 'rgba(148, 163, 184, 0.12)',
              },
            },
            '& .MuiListItemIcon-root': {
              color: '#e2e8f0',
              minWidth: '32px',
            },
            '& .MuiListItemText-primary': {
              fontFamily,
              color: '#e2e8f0',
            },
          },
        }}
        MenuListProps={{
          sx: {
            fontFamily,
            '& .MuiMenuItem-root': {
              fontFamily,
            },
            '& .MuiListItemText-primary': {
              fontFamily,
            },
          },
        }}
      >
        <MenuItem onClick={() => handleShare('twitter')}>
          <ListItemIcon>
            <Icon>chat</Icon>
          </ListItemIcon>
          <ListItemText>Twitter</ListItemText>
        </MenuItem>

        <MenuItem onClick={() => handleShare('facebook')}>
          <ListItemIcon>
            <Icon>facebook</Icon>
          </ListItemIcon>
          <ListItemText>Facebook</ListItemText>
        </MenuItem>

        <MenuItem onClick={() => handleShare('linkedin')}>
          <ListItemIcon>
            <Icon>business</Icon>
          </ListItemIcon>
          <ListItemText>LinkedIn</ListItemText>
        </MenuItem>

        <MenuItem onClick={() => handleShare('reddit')}>
          <ListItemIcon>
            <Icon>forum</Icon>
          </ListItemIcon>
          <ListItemText>Reddit</ListItemText>
        </MenuItem>

        <MenuItem onClick={handleCopyLink}>
          <ListItemIcon>
            <Icon>link</Icon>
          </ListItemIcon>
          <ListItemText>Copy Link</ListItemText>
        </MenuItem>

        <MenuItem onClick={handleDownloadCard}>
          <ListItemIcon>
            <Icon>download</Icon>
          </ListItemIcon>
          <ListItemText>Download Card</ListItemText>
        </MenuItem>
      </Menu>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
}

export default ShareButton;
