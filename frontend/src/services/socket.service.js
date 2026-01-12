/**
 * WebSocket Service for Material Dashboard PRO Integration
 * Place this file in: frontend/src/services/socket.service.js
 */

import io from 'socket.io-client';
import config from '../config';

class SocketService {
  constructor() {
    this.socket = null;
    this.connected = false;
    this.listeners = new Map();
  }

  /**
   * Connect to WebSocket server
   * @returns {Socket} Socket instance
   */
  connect() {
    if (this.socket) {
      console.warn('Socket already connected');
      return this.socket;
    }

    const transports = ['polling'];

    console.log('Connecting to WebSocket:', config.WS_URL);
    
    this.socket = io(config.WS_URL, {
      transports,
      upgrade: false,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    // Connection events
    this.socket.on('connect', () => {
      console.log('✓ WebSocket connected');
      this.connected = true;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('✗ WebSocket disconnected:', reason);
      this.connected = false;
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
    });

    return this.socket;
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.socket) {
      console.log('Disconnecting WebSocket');
      this.socket.disconnect();
      this.socket = null;
      this.connected = false;
      this.listeners.clear();
    }
  }

  /**
   * Check if connected
   * @returns {boolean}
   */
  isConnected() {
    return this.connected && this.socket !== null;
  }

  /**
   * Listen to an event
   * @param {string} event - Event name
   * @param {function} callback - Callback function
   */
  on(event, callback) {
    if (!this.socket) {
      console.error('Socket not connected. Call connect() first.');
      return;
    }

    this.socket.on(event, callback);
    
    // Track listeners for cleanup
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {function} callback - Optional specific callback to remove
   */
  off(event, callback) {
    if (!this.socket) return;

    if (callback) {
      this.socket.off(event, callback);
      
      // Remove from tracking
      const callbacks = this.listeners.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    } else {
      // Remove all listeners for this event
      this.socket.off(event);
      this.listeners.delete(event);
    }
  }

  /**
   * Emit an event to server
   * @param {string} event - Event name
   * @param {*} data - Data to send
   */
  emit(event, data) {
    if (!this.socket) {
      console.error('Socket not connected. Cannot emit:', event);
      return;
    }

    console.log('→ Emitting:', event, data);
    this.socket.emit(event, data);
  }

  /**
   * Listen to event once
   * @param {string} event - Event name
   * @param {function} callback - Callback function
   */
  once(event, callback) {
    if (!this.socket) {
      console.error('Socket not connected. Call connect() first.');
      return;
    }

    this.socket.once(event, callback);
  }

  /**
   * Get socket instance (for advanced usage)
   * @returns {Socket}
   */
  getSocket() {
    return this.socket;
  }
}

// Export singleton instance
export default new SocketService();
