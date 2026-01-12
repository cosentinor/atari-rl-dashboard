/**
 * Analytics Service
 * Tracks user interactions and sends to backend
 */

import config from '../config';

class AnalyticsService {
  constructor() {
    this.visitorId = null;
    this.eventQueue = [];
    this.flushInterval = null;
  }

  initialize() {
    // Get or create visitor ID
    this.visitorId = localStorage.getItem('visitor_id');
    if (!this.visitorId) {
      this.visitorId = this.generateVisitorId();
      localStorage.setItem('visitor_id', this.visitorId);
    }

    // Register visitor
    this.registerVisitor();

    // Start auto-flush (every 30 seconds)
    this.flushInterval = setInterval(() => this.flush(), 30000);

    // Track page view
    this.trackPageView();
  }

  generateVisitorId() {
    return 'visitor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  async registerVisitor() {
    try {
      await fetch(`${config.API_BASE_URL}/api/analytics/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          visitor_id: this.visitorId,
          user_agent: navigator.userAgent,
          screen_resolution: `${window.screen.width}x${window.screen.height}`,
          referrer: document.referrer || 'direct'
        })
      });
    } catch (err) {
      console.error('Failed to register visitor:', err);
    }
  }

  trackEvent(eventType, eventData = {}) {
    const event = {
      visitor_id: this.visitorId,
      event_type: eventType,
      event_data: eventData,
      timestamp: new Date().toISOString()
    };

    this.eventQueue.push(event);

    // Flush if queue is large
    if (this.eventQueue.length >= 10) {
      this.flush();
    }
  }

  trackPageView() {
    this.trackEvent('page_view', {
      path: window.location.pathname,
      url: window.location.href
    });
  }

  trackTrainingStart(gameId) {
    this.trackEvent('training_start', { game_id: gameId });
  }

  trackTrainingStop(gameId, episodes, bestReward) {
    this.trackEvent('training_stop', {
      game_id: gameId,
      episodes,
      best_reward: bestReward
    });
  }

  trackShare(platform, sessionId) {
    this.trackEvent('share', { platform, session_id: sessionId });
  }

  trackFeedbackSubmit(category, rating) {
    this.trackEvent('feedback_submit', { category, rating });
  }

  trackModelSave(gameId, episode, reward) {
    this.trackEvent('model_save', {
      game_id: gameId,
      episode,
      reward
    });
  }

  trackSpeedChange(speed) {
    this.trackEvent('speed_change', { speed });
  }

  async flush() {
    if (this.eventQueue.length === 0) return;

    const events = [...this.eventQueue];
    this.eventQueue = [];

    try {
      await fetch(`${config.API_BASE_URL}/api/analytics/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ events })
      });
    } catch (err) {
      console.error('Failed to send analytics:', err);
      // Re-queue events if failed
      this.eventQueue = [...events, ...this.eventQueue];
    }
  }

  cleanup() {
    // Flush remaining events
    this.flush();
    
    // Clear interval
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
  }
}

// Export singleton
const analyticsService = new AnalyticsService();
export default analyticsService;
