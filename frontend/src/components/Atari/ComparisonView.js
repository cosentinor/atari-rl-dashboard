/**
 * Comparison View Component
 * Compare multiple model checkpoints side-by-side
 */

import { useState, useEffect } from 'react';
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardActionArea from "@mui/material/CardActionArea";
import Checkbox from "@mui/material/Checkbox";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import CircularProgress from "@mui/material/CircularProgress";
import Icon from "@mui/material/Icon";
import IconButton from "@mui/material/IconButton";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import MDButton from "components/MDButton";
import MDBadge from "components/MDBadge";
import config from "config";

function ComparisonView({ open, onClose, gameId }) {
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open && gameId) {
      fetchCheckpoints();
    }
  }, [open, gameId]);

  const fetchCheckpoints = async () => {
    try {
      const gameKey = gameId.replace(/\//g, '_');
      const response = await fetch(`${config.API_BASE_URL}/api/models/${gameKey}`);
      const data = await response.json();
      if (data.success) {
        setCheckpoints(data.checkpoints || []);
      }
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
    }
  };

  const handleModelToggle = (checkpoint) => {
    setSelectedModels(prev => {
      const exists = prev.find(m => m.filename === checkpoint.filename);
      if (exists) {
        return prev.filter(m => m.filename !== checkpoint.filename);
      } else if (prev.length < 3) {
        return [...prev, checkpoint];
      }
      return prev;
    });
  };

  const handleCompare = async () => {
    if (selectedModels.length < 2) return;

    setLoading(true);
    try {
      const response = await fetch(`${config.API_BASE_URL}/api/models/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: gameId,
          checkpoints: selectedModels.map(m => m.filename)
        })
      });

      const data = await response.json();
      if (data.success) {
        setComparisonData(data.comparison);
      }
    } catch (err) {
      console.error('Failed to compare models:', err);
    } finally {
      setLoading(false);
    }
  };

  const calculateImprovement = () => {
    if (selectedModels.length < 2) return null;
    const sorted = [...selectedModels].sort((a, b) => a.episode - b.episode);
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    const improvement = ((last.reward - first.reward) / first.reward) * 100;
    return improvement.toFixed(1);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth={false}
      PaperProps={{
        sx: {
          minHeight: '70vh',
          width: '520px',
          maxWidth: '92vw',
          background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
          border: '1px solid rgba(148, 163, 184, 0.2)',
          borderRadius: '16px',
          boxShadow: '0 20px 40px rgba(0,0,0,0.45)',
          color: '#e2e8f0',
        }
      }}
    >
      <DialogTitle>
        <MDBox display="flex" justifyContent="space-between" alignItems="center">
          <MDBox display="flex" alignItems="center" gap={1}>
            <Icon sx={{ color: '#0ea5e9' }}>compare_arrows</Icon>
            <MDTypography variant="h6" fontWeight="medium">
              Compare Checkpoints
            </MDTypography>
          </MDBox>
          <IconButton onClick={onClose} size="small">
            <Icon>close</Icon>
          </IconButton>
        </MDBox>
      </DialogTitle>

      <DialogContent>
        {!comparisonData ? (
          <MDBox>
            <MDTypography variant="h6" mb={2} color="white">
              Select 2-3 checkpoints to compare
            </MDTypography>

            <Grid container spacing={2}>
              {checkpoints.map((cp) => {
                const isSelected = selectedModels.find(m => m.filename === cp.filename);
                
                return (
                  <Grid item xs={12} sm={6} md={4} key={cp.filename}>
                    <Card
                      sx={{
                        border: isSelected ? '2px solid' : '1px solid',
                        borderColor: isSelected ? 'info.main' : 'divider',
                        backgroundColor: isSelected ? 'rgba(6, 182, 212, 0.08)' : 'rgba(15, 23, 42, 0.4)',
                        boxShadow: '0 10px 18px rgba(0,0,0,0.35)',
                        borderRadius: '12px',
                      }}
                    >
                      <CardActionArea
                        onClick={() => handleModelToggle(cp)}
                        disabled={!isSelected && selectedModels.length >= 3}
                      >
                        <CardContent>
                          <MDBox display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                            <MDTypography variant="button" fontWeight="bold" color="white">
                              Episode {cp.episode}
                            </MDTypography>
                            {cp.is_best && <Icon sx={{ color: '#facc15' }}>star</Icon>}
                            <Checkbox checked={!!isSelected} color="info" />
                          </MDBox>
                          
                          <MDTypography variant="h6" color="info" mb={1}>
                            {cp.reward?.toFixed(1)}
                          </MDTypography>
                          
                          <MDTypography variant="caption" color="text">
                            {new Date(cp.timestamp).toLocaleDateString()}
                          </MDTypography>
                        </CardContent>
                      </CardActionArea>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>

            {selectedModels.length >= 2 && (
              <MDBox mt={3} textAlign="center">
                <MDBadge
                  badgeContent={`${selectedModels.length} selected`}
                  color="info"
                  container
                  sx={{ mb: 2 }}
                />
                
                {calculateImprovement() && (
                <MDTypography variant="body2" color="success" mb={2}>
                  Improvement: +{calculateImprovement()}%
                </MDTypography>
                )}
                
                <MDButton
                  variant="gradient"
                  color="info"
                  onClick={handleCompare}
                  disabled={loading}
                >
                  {loading ? 'Comparing...' : 'Compare Selected'}
                </MDButton>
              </MDBox>
            )}
          </MDBox>
        ) : (
          <MDBox>
            <MDButton
              variant="text"
              color="secondary"
              onClick={() => setComparisonData(null)}
              startIcon={<Icon>arrow_back</Icon>}
              sx={{ mb: 2 }}
            >
              Back to Selection
            </MDButton>

            <MDTypography variant="h6" mb={3} color="white">
              Comparison Results
            </MDTypography>

            <TableContainer sx={{ backgroundColor: 'rgba(15, 23, 42, 0.4)', borderRadius: '12px' }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: '#e2e8f0' }}>Metric</TableCell>
                    {selectedModels.map((model, i) => (
                      <TableCell key={i} align="right" sx={{ color: '#e2e8f0' }}>
                        Episode {model.episode}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell sx={{ color: '#cbd5f5' }}>Best Reward</TableCell>
                    {selectedModels.map((model, i) => (
                      <TableCell key={i} align="right">
                        <MDTypography variant="button" fontWeight="bold" color="warning">
                          {model.reward?.toFixed(1)}
                        </MDTypography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ color: '#cbd5f5' }}>Episode</TableCell>
                    {selectedModels.map((model, i) => (
                      <TableCell key={i} align="right" sx={{ color: '#e2e8f0' }}>{model.episode}</TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ color: '#cbd5f5' }}>Date</TableCell>
                    {selectedModels.map((model, i) => (
                      <TableCell key={i} align="right" sx={{ color: '#e2e8f0' }}>
                        {new Date(model.timestamp).toLocaleDateString()}
                      </TableCell>
                    ))}
                  </TableRow>
                  {selectedModels[0].is_best && (
                    <TableRow>
                      <TableCell sx={{ color: '#cbd5f5' }}>Status</TableCell>
                      {selectedModels.map((model, i) => (
                        <TableCell key={i} align="right" sx={{ color: '#e2e8f0' }}>
                          {model.is_best ? 'Best' : '-'}
                        </TableCell>
                      ))}
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>

            {calculateImprovement() && (
              <MDBox mt={3} textAlign="center">
                <MDBadge
                  badgeContent={`+${calculateImprovement()}% Improvement`}
                  color="success"
                  size="lg"
                  container
                />
              </MDBox>
            )}
          </MDBox>
        )}
      </DialogContent>

      <DialogActions>
        <MDButton variant="text" color="secondary" onClick={onClose}>
          Close
        </MDButton>
      </DialogActions>
    </Dialog>
  );
}

export default ComparisonView;
