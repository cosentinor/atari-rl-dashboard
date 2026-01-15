/**
 * GameCanvas Component - Atari RL Training Dashboard
 * Displays real-time game frames from training
 */

import { useRef, useEffect, useCallback, useState } from 'react';
import Card from "@mui/material/Card";
import Icon from "@mui/material/Icon";
import MDBox from "components/MDBox";
import MDTypography from "components/MDTypography";
import ShareButton from "components/Atari/ShareButton";

function GameCanvas({ isTraining, sessionId, selectedGame, stats }) {
  const canvasRef = useRef(null);
  const frameBufferRef = useRef([]);
  const renderingRef = useRef(false);
  const frameImageRef = useRef(new Image());
  const animationFrameRef = useRef(null);
  
  const [fps, setFps] = useState(0);
  const fpsRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  const renderFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      renderingRef.current = false;
      return;
    }
    
    try {
      const ctx = canvas.getContext('2d');
      const buffer = frameBufferRef.current;
      
      if (buffer.length > 0) {
        const frameData = buffer.shift();
        
        frameImageRef.current.onload = null;
        frameImageRef.current.onerror = null;
        
        frameImageRef.current.onload = () => {
          try {
            ctx.drawImage(frameImageRef.current, 0, 0, canvas.width, canvas.height);
          } catch (err) {
            console.warn('Canvas draw error:', err);
          }
        };
        
        frameImageRef.current.onerror = (err) => {
          console.warn('Frame image load error:', err);
        };
        
        frameImageRef.current.src = frameData;
        
        // Update FPS
        frameCountRef.current++;
        const now = Date.now();
        if (now - lastFpsUpdateRef.current >= 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }
      }
      
      if (isTraining || buffer.length > 0) {
        animationFrameRef.current = requestAnimationFrame(renderFrame);
      } else {
        renderingRef.current = false;
      }
    } catch (err) {
      console.error('Render frame error:', err);
      if (isTraining) {
        animationFrameRef.current = requestAnimationFrame(renderFrame);
      } else {
        renderingRef.current = false;
      }
    }
  }, [isTraining]);

  const startRenderLoop = useCallback(() => {
    if (!renderingRef.current && canvasRef.current) {
      renderingRef.current = true;
      animationFrameRef.current = requestAnimationFrame(renderFrame);
    }
  }, [renderFrame]);

  // Clear canvas when training stops
  useEffect(() => {
    if (!isTraining && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      frameBufferRef.current = [];
    }
  }, [isTraining]);

  // Start render loop when training starts
  useEffect(() => {
    if (isTraining) {
      startRenderLoop();
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isTraining, startRenderLoop]);

  // Setup global frame handler
  useEffect(() => {
    window.addFrame = (frameData) => {
      if (frameBufferRef.current.length < 30) {
        frameBufferRef.current.push(frameData);
      } else {
        frameBufferRef.current.shift();
        frameBufferRef.current.push(frameData);
      }
      startRenderLoop();
    };
    
    window.clearCanvas = () => {
      frameBufferRef.current = [];
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    };
    
    return () => {
      delete window.addFrame;
      delete window.clearCanvas;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [startRenderLoop]);

  return (
    <Card
      sx={{
        background: 'linear-gradient(145deg, #0f1628 0%, #0b1224 100%)',
        border: '1px solid rgba(148, 163, 184, 0.18)',
        boxShadow: '0 16px 36px rgba(0, 0, 0, 0.45)',
        borderRadius: '16px',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <MDBox p={2} sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <MDBox display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <MDTypography variant="h6" fontWeight="medium" display="flex" alignItems="center" gap={0.75}>
            <Icon sx={{ fontSize: '1.3rem !important', color: '#0ea5e9' }}>sports_esports</Icon>
            Live Game {isTraining && <span style={{ color: '#4caf50', fontSize: '0.8rem' }}>● TRAINING</span>}
          </MDTypography>
          <MDBox display="flex" alignItems="center" gap={1}>
            {isTraining && (
              <MDTypography variant="caption" color="text" fontWeight="medium">
                {fps} FPS
              </MDTypography>
            )}
            <ShareButton
              sessionId={sessionId}
              gameId={selectedGame}
              bestReward={stats?.bestReward}
              episodes={stats?.episode}
            />
          </MDBox>
        </MDBox>
        
        <MDBox
          sx={{
            position: 'relative',
            background: '#000',
            borderRadius: '12px',
            overflow: 'hidden',
            border: isTraining ? '3px solid #4caf50' : '3px solid rgba(255,255,255,0.1)',
            boxShadow: isTraining ? '0 0 30px rgba(76, 175, 80, 0.4)' : '0 4px 20px rgba(0,0,0,0.3)',
            transition: 'all 0.3s',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: { xs: '240px', sm: '300px', md: '340px', lg: '380px' },
            maxHeight: '420px',
          }}
        >
          <canvas 
            ref={canvasRef} 
            width={640} 
            height={480}
            style={{
              display: 'block',
              width: 'auto',
                    maxWidth: '100%',
              maxHeight: '100%',
              imageRendering: 'pixelated',
            }}
          />
          {!isTraining && (
            <MDBox
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                backdropFilter: 'blur(8px)',
              }}
            >
              <MDTypography variant="h1" color="white" mb={1} sx={{ fontSize: '4rem' }}>
                <Icon sx={{ fontSize: '3.2rem !important', color: '#0ea5e9' }}>sports_esports</Icon>
              </MDTypography>
              <MDTypography variant="h6" color="white" fontWeight="medium">
                Select a game and click Start
              </MDTypography>
              <MDTypography variant="caption" color="white" sx={{ opacity: 0.6, mt: 1 }}>
                Watch AI learn to play classic Atari games
              </MDTypography>
            </MDBox>
          )}
        </MDBox>
      </MDBox>
    </Card>
  );
}

export default GameCanvas;
