/**
 * Feedback Widget Component
 * Floating feedback button with quick rating and detailed form
 */

import { useState } from 'react';
import Fab from "@mui/material/Fab";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Rating from "@mui/material/Rating";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Icon from "@mui/material/Icon";
import IconButton from "@mui/material/IconButton";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import MDButton from "components/MDButton";
import { PODCAST } from "constants/podcast";

function FeedbackWidget() {
  const [open, setOpen] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [rating, setRating] = useState(0);
  const [category, setCategory] = useState('general');
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [podcastLogoError, setPodcastLogoError] = useState(false);

  const ratingSx = {
    color: '#fbbf24',
    '& .MuiRating-iconEmpty': {
      color: 'rgba(226, 232, 240, 0.55)',
    },
  };

  const handleQuickRating = async (stars) => {
    setRating(stars);
    
    const visitorId = localStorage.getItem('visitor_id');
    
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          visitor_id: visitorId,
          category: 'quick_rating',
          rating: stars,
          message: null
        })
      });

      setSnackbar({ open: true, message: 'Thanks for your feedback!', severity: 'success' });
      setTimeout(() => {
        setOpen(false);
        setRating(0);
      }, 2000);
    } catch (err) {
      console.error('Failed to submit rating:', err);
      setSnackbar({ open: true, message: 'Failed to submit feedback', severity: 'error' });
    }
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    const visitorId = localStorage.getItem('visitor_id');

    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          visitor_id: visitorId,
          category,
          rating,
          message,
          email: email || null
        })
      });

      setSnackbar({ open: true, message: 'Feedback submitted successfully!', severity: 'success' });
      setOpen(false);
      setShowForm(false);
      setRating(0);
      setMessage('');
      setEmail('');
      setCategory('general');
    } catch (err) {
      console.error('Failed to submit feedback:', err);
      setSnackbar({ open: true, message: 'Failed to submit feedback', severity: 'error' });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      {/* Floating Action Button */}
      <Fab
        color="info"
        aria-label="feedback"
        onClick={() => setOpen(true)}
        size="small"
        sx={{
          position: 'fixed',
          bottom: 20,
          left: 20,
          zIndex: 1500,
          backgroundColor: '#0ea5e9',
          '&:hover': { backgroundColor: '#0284c7' },
        }}
      >
        <Icon sx={{ fontSize: '1.2rem !important' }}>feedback</Icon>
      </Fab>

      {/* Feedback Dialog */}
      <Dialog
        open={open}
        onClose={() => {
          setOpen(false);
          setShowForm(false);
        }}
        maxWidth="xs"
        fullWidth
        PaperProps={{
          sx: {
            backgroundColor: '#0b1224',
            color: '#e2e8f0',
            border: '1px solid rgba(148, 163, 184, 0.25)',
            borderRadius: '16px',
          }
        }}
      >
        <DialogTitle>
          <MDBox display="flex" justifyContent="space-between" alignItems="center">
            <MDBox display="flex" alignItems="center" gap={1}>
              <Icon sx={{ color: '#0ea5e9' }}>feedback</Icon>
              <MDTypography variant="h5" fontWeight="medium">
                Feedback
              </MDTypography>
            </MDBox>
            <IconButton onClick={() => setOpen(false)} size="small">
              <Icon>close</Icon>
            </IconButton>
          </MDBox>
        </DialogTitle>

        <DialogContent>
          {!showForm ? (
            <MDBox textAlign="center" py={2}>
              <MDTypography variant="h6" mb={2}>
                How would you rate your experience?
              </MDTypography>
              
              <MDBox display="flex" justifyContent="center" mb={3}>
                <Rating
                  name="quick-rating"
                  value={rating}
                  onChange={(event, newValue) => {
                    if (newValue) {
                      handleQuickRating(newValue);
                    }
                  }}
                  size="large"
                  sx={{ ...ratingSx, fontSize: '3rem' }}
                />
              </MDBox>

              <MDTypography variant="body2" color="text" mb={3}>
                Or provide detailed feedback
              </MDTypography>

              <MDButton
                variant="outlined"
                color="info"
                onClick={() => setShowForm(true)}
                fullWidth
              >
                Write Detailed Feedback
              </MDButton>
            </MDBox>
          ) : (
            <form onSubmit={handleFormSubmit}>
              <MDBox display="flex" flexDirection="column" gap={2}>
                <FormControl fullWidth>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    label="Category"
                  >
                    <MenuItem value="general">General Feedback</MenuItem>
                    <MenuItem value="bug">Bug Report</MenuItem>
                    <MenuItem value="feature">Feature Request</MenuItem>
                    <MenuItem value="ui">UI/UX Feedback</MenuItem>
                    <MenuItem value="performance">Performance Issue</MenuItem>
                  </Select>
                </FormControl>

                <MDBox>
                  <MDTypography variant="caption" color="text" mb={1} display="block">
                    Rating (optional)
                  </MDTypography>
                  <Rating
                    name="form-rating"
                    value={rating}
                    onChange={(event, newValue) => setRating(newValue)}
                    sx={ratingSx}
                  />
                </MDBox>

                <TextField
                  label="Your Feedback"
                  multiline
                  rows={4}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  required
                  fullWidth
                  placeholder="Tell us what you think..."
                />

                <TextField
                  label="Email (optional)"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  fullWidth
                  placeholder="your@email.com"
                  helperText="We'll only use this to follow up on your feedback"
                />
              </MDBox>
            </form>
          )}

          <MDBox
            mt={3}
            pt={2}
            sx={{
              borderTop: '1px solid rgba(148, 163, 184, 0.2)',
            }}
          >
            <MDBox
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              flexWrap="nowrap"
              gap={1}
              px={2}
              py={1.5}
              borderRadius="14px"
              sx={{
                background: 'linear-gradient(135deg, #131a2c 0%, #0d1424 100%)',
                border: '1px solid rgba(148, 163, 184, 0.25)',
              }}
            >
              <MDBox
                component="a"
                href={PODCAST.url}
                target="_blank"
                rel="noopener noreferrer"
                display="inline-flex"
                alignItems="center"
                gap={1}
                sx={{
                  textDecoration: 'none',
                  color: 'inherit',
                  flex: '1 1 auto',
                  minWidth: 0,
                }}
                aria-label={`Subscribe to ${PODCAST.name} podcast`}
              >
                <MDBox
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  width="48px"
                  height="48px"
                  borderRadius="12px"
                  sx={{
                    backgroundColor: '#0b1224',
                    border: '1px solid rgba(148, 163, 184, 0.35)',
                    overflow: 'hidden',
                    flexShrink: 0,
                  }}
                >
                  {!podcastLogoError ? (
                    <img
                      src={PODCAST.logoSrc}
                      alt={PODCAST.logoAlt}
                      onError={() => setPodcastLogoError(true)}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  ) : (
                    <MDTypography variant="caption" fontWeight="bold" sx={{ color: '#e2e8f0' }}>
                      {PODCAST.shortName}
                    </MDTypography>
                  )}
                </MDBox>
                <MDBox display="flex" flexDirection="column">
                  <MDTypography
                    variant="button"
                    fontWeight="bold"
                    sx={{
                      background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      lineHeight: 1.1,
                      fontSize: '0.92rem',
                    }}
                  >
                    Major Programmes
                  </MDTypography>
                  <MDTypography
                    variant="button"
                    fontWeight="bold"
                    sx={{
                      background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      lineHeight: 1.1,
                      fontSize: '0.92rem',
                    }}
                  >
                    Navigating
                  </MDTypography>
                </MDBox>
              </MDBox>
              <MDButton
                component="a"
                href={PODCAST.url}
                target="_blank"
                rel="noopener noreferrer"
                variant="contained"
                color="info"
                size="small"
                sx={{
                  textTransform: 'none',
                  borderRadius: '12px',
                  width: '104px',
                  height: '48px',
                  minWidth: '104px',
                  minHeight: '48px',
                  p: 0,
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  background: 'linear-gradient(180deg, #0ea5e9 0%, #8b5cf6 100%)',
                  color: '#fff',
                  border: '1px solid rgba(14, 165, 233, 0.35)',
                  boxShadow: '0 10px 20px rgba(14, 165, 233, 0.25)',
                  '&:hover': {
                    background: 'linear-gradient(180deg, #22d3ee 0%, #6366f1 100%)',
                  },
                }}
              >
                Subscribe
              </MDButton>
            </MDBox>
          </MDBox>
        </DialogContent>

        {showForm && (
          <DialogActions>
            <MDButton
              variant="text"
              color="secondary"
              onClick={() => setShowForm(false)}
            >
              Back
            </MDButton>
            <MDButton
              variant="gradient"
              color="info"
              onClick={handleFormSubmit}
              disabled={isSubmitting || !message}
            >
              {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
            </MDButton>
          </DialogActions>
        )}
      </Dialog>

      {/* Success/Error Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
}

export default FeedbackWidget;
