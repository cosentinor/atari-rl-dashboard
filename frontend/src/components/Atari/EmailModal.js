/**
 * Email Modal Component
 * Custom modal without MUI Dialog - guaranteed dark mode styling
 */

import { useState } from 'react';
import ReactDOM from 'react-dom';
import config from "config";

function EmailModal({ open, onClose }) {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  };

  const handleSubmit = async () => {
    if (!email || !validateEmail(email)) {
      setError('Please enter a valid email address');
      return;
    }

    setIsSubmitting(true);
    const visitorId = localStorage.getItem('visitor_id');

    try {
      await fetch(`${config.API_BASE_URL}/api/analytics/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          visitor_id: visitorId,
          email: email,
          email_collected: true
        })
      });

      localStorage.setItem('email_collected', 'true');
      localStorage.setItem('user_email', email);
      onClose();
    } catch (err) {
      console.error('Failed to submit email:', err);
      setError('Failed to submit. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAnonymous = () => {
    localStorage.setItem('email_collected', 'anonymous');
    onClose();
  };

  if (!open) return null;

  const modalContent = (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.75)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      backdropFilter: 'blur(4px)'
    }} onClick={handleAnonymous}>
      <div style={{
        backgroundColor: '#1a2035',
        borderRadius: '16px',
        border: '1px solid rgba(255, 255, 255, 0.15)',
        boxShadow: '0 25px 80px rgba(0, 0, 0, 0.6)',
        width: '90%',
        maxWidth: '480px',
        maxHeight: '90vh',
        overflow: 'auto',
        color: '#ffffff'
      }} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={{
          padding: '20px 24px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span style={{ fontSize: '1.25rem', fontWeight: 600, color: '#ffffff' }}>
            🎉 Great Progress!
          </span>
          <button 
            onClick={handleAnonymous}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'rgba(255, 255, 255, 0.6)',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: '4px'
            }}
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: '24px' }}>
          <p style={{ 
            color: 'rgba(255, 255, 255, 0.9)', 
            marginBottom: '24px', 
            fontSize: '0.95rem',
            lineHeight: 1.5
          }}>
            You've completed your first training session! Want to save your progress and get updates?
          </p>

          {/* Email Input */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{ 
              display: 'block', 
              color: 'rgba(255, 255, 255, 0.7)', 
              fontSize: '0.8rem',
              marginBottom: '8px',
              fontWeight: 500
            }}>
              Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                setError('');
              }}
              placeholder="your@email.com"
              style={{
                width: '100%',
                padding: '12px 16px',
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
                border: error ? '2px solid #ef4444' : '2px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '1rem',
                outline: 'none',
                boxSizing: 'border-box'
              }}
              onFocus={(e) => {
                e.target.style.borderColor = '#06b6d4';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = error ? '#ef4444' : 'rgba(255, 255, 255, 0.2)';
              }}
            />
            {error && (
              <p style={{ color: '#ef4444', fontSize: '0.8rem', marginTop: '8px' }}>
                {error}
              </p>
            )}
          </div>

          {/* Benefits */}
          <p style={{ 
            color: '#ffffff', 
            fontWeight: 600, 
            marginBottom: '16px',
            fontSize: '0.95rem'
          }}>
            Benefits:
          </p>

          <div style={{ marginBottom: '20px' }}>
            {[
              { title: 'Save your models', desc: 'Access your trained models anytime' },
              { title: 'Track your progress', desc: 'See your learning journey and improvements' },
              { title: 'Get tips & updates', desc: 'Receive training strategies and new features' }
            ].map((item, i) => (
              <div key={i} style={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                gap: '12px', 
                marginBottom: '14px' 
              }}>
                <span style={{ color: '#22c55e', fontSize: '1.2rem', lineHeight: 1 }}>✓</span>
                <div>
                  <p style={{ 
                    color: '#ffffff', 
                    margin: 0, 
                    fontWeight: 500, 
                    fontSize: '0.9rem',
                    lineHeight: 1.3
                  }}>
                    {item.title}
                  </p>
                  <p style={{ 
                    color: 'rgba(255, 255, 255, 0.5)', 
                    margin: '4px 0 0 0', 
                    fontSize: '0.8rem' 
                  }}>
                    {item.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <p style={{ 
            color: 'rgba(255, 255, 255, 0.4)', 
            textAlign: 'center', 
            fontSize: '0.75rem',
            marginBottom: '8px'
          }}>
            🔒 We respect your privacy. No spam, unsubscribe anytime.
          </p>
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 24px',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '12px'
        }}>
          <button
            onClick={handleAnonymous}
            style={{
              padding: '10px 20px',
              backgroundColor: 'transparent',
              color: 'rgba(255, 255, 255, 0.7)',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '0.9rem',
              fontWeight: 500
            }}
          >
            Continue Anonymously
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || !email}
            style={{
              padding: '10px 24px',
              background: 'linear-gradient(195deg, #49a3f1, #1A73E8)',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              cursor: isSubmitting || !email ? 'not-allowed' : 'pointer',
              fontSize: '0.9rem',
              fontWeight: 600,
              opacity: isSubmitting || !email ? 0.5 : 1
            }}
          >
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </button>
        </div>
      </div>
    </div>
  );

  // Use portal to render outside the component tree
  return ReactDOM.createPortal(modalContent, document.body);
}

export default EmailModal;
