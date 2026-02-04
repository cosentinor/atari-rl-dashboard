#!/usr/bin/env python3
"""
Daily Metrics Report Generator for Atari RL Dashboard.
Collects website traffic, user engagement, and training statistics.
Sends a formatted email report.
"""

import os
import sys
import sqlite3
import smtplib
import subprocess
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration
DB_PATH = Path(__file__).parent / "data" / "rl_training.db"
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "alerts@riccardocosentino.com")
SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "25"))
HOSTNAME = subprocess.getoutput("hostname")


class MetricsCollector:
    """Collect metrics from the database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_visitor_metrics(self, days: int = 1) -> Dict[str, Any]:
        """Get visitor metrics for the last N days."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        metrics = {}
        
        # Total visitors
        cursor.execute("SELECT COUNT(*) as total FROM visitors")
        metrics['total_visitors'] = cursor.fetchone()['total']
        
        # Visitors with email
        cursor.execute("SELECT COUNT(*) as count FROM visitors WHERE email IS NOT NULL")
        metrics['visitors_with_email'] = cursor.fetchone()['count']
        
        # New visitors today
        cursor.execute("""
            SELECT COUNT(*) as count FROM visitors
            WHERE DATE(first_visit) = DATE('now')
        """)
        metrics['new_visitors_today'] = cursor.fetchone()['count']
        
        # New visitors yesterday
        cursor.execute("""
            SELECT COUNT(*) as count FROM visitors
            WHERE DATE(first_visit) = DATE('now', '-1 day')
        """)
        metrics['new_visitors_yesterday'] = cursor.fetchone()['count']
        
        # Returning visitors today
        cursor.execute("""
            SELECT COUNT(*) as count FROM visitors
            WHERE DATE(last_visit) = DATE('now')
            AND DATE(first_visit) < DATE('now')
        """)
        metrics['returning_visitors_today'] = cursor.fetchone()['count']
        
        # Marketing opt-ins
        cursor.execute("SELECT COUNT(*) as count FROM visitors WHERE opt_in_marketing = 1")
        metrics['marketing_opt_ins'] = cursor.fetchone()['count']
        
        # New visitors last 7 days (trend)
        cursor.execute("""
            SELECT DATE(first_visit) as date, COUNT(*) as count
            FROM visitors
            WHERE first_visit >= DATE('now', '-7 days')
            GROUP BY DATE(first_visit)
            ORDER BY date
        """)
        metrics['visitors_trend_7d'] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return metrics
    
    def get_session_metrics(self, days: int = 1) -> Dict[str, Any]:
        """Get training session metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        metrics = {}
        
        # Total sessions all time
        cursor.execute("SELECT COUNT(*) as total FROM sessions")
        metrics['total_sessions'] = cursor.fetchone()['total']
        
        # Sessions today
        cursor.execute("""
            SELECT COUNT(*) as count FROM sessions
            WHERE DATE(started_at) = DATE('now')
        """)
        metrics['sessions_today'] = cursor.fetchone()['count']
        
        # Sessions yesterday
        cursor.execute("""
            SELECT COUNT(*) as count FROM sessions
            WHERE DATE(started_at) = DATE('now', '-1 day')
        """)
        metrics['sessions_yesterday'] = cursor.fetchone()['count']
        
        # Sessions by game (today)
        cursor.execute("""
            SELECT game_id, COUNT(*) as count
            FROM sessions
            WHERE DATE(started_at) = DATE('now')
            GROUP BY game_id
            ORDER BY count DESC
        """)
        metrics['sessions_by_game_today'] = {row['game_id']: row['count'] for row in cursor.fetchall()}
        
        # Average session duration today (in minutes)
        cursor.execute("""
            SELECT AVG((julianday(ended_at) - julianday(started_at)) * 24 * 60) as avg_minutes
            FROM sessions
            WHERE DATE(started_at) = DATE('now')
            AND ended_at IS NOT NULL
        """)
        result = cursor.fetchone()
        metrics['avg_session_duration_minutes'] = round(result['avg_minutes'], 1) if result['avg_minutes'] else 0
        
        # Total episodes today
        cursor.execute("""
            SELECT SUM(total_episodes) as total
            FROM sessions
            WHERE DATE(started_at) = DATE('now')
        """)
        result = cursor.fetchone()
        metrics['total_episodes_today'] = result['total'] or 0
        
        conn.close()
        return metrics
    
    def get_analytics_metrics(self, days: int = 1) -> Dict[str, Any]:
        """Get analytics event metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        metrics = {}
        
        # Event counts today
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM analytics_events
            WHERE DATE(timestamp) = DATE('now')
            GROUP BY event_type
            ORDER BY count DESC
        """)
        metrics['events_today'] = {row['event_type']: row['count'] for row in cursor.fetchall()}
        
        # Event counts yesterday
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM analytics_events
            WHERE DATE(timestamp) = DATE('now', '-1 day')
            GROUP BY event_type
            ORDER BY count DESC
        """)
        metrics['events_yesterday'] = {row['event_type']: row['count'] for row in cursor.fetchall()}
        
        # Page views trend (7 days)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM analytics_events
            WHERE event_type = 'page_view'
            AND timestamp >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
        metrics['page_views_trend_7d'] = [dict(row) for row in cursor.fetchall()]
        
        # Conversion funnel
        cursor.execute("SELECT COUNT(DISTINCT visitor_id) as count FROM analytics_events WHERE event_type = 'page_view'")
        funnel_visitors = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(DISTINCT visitor_id) as count FROM analytics_events WHERE event_type = 'email_provided'")
        funnel_email = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(DISTINCT visitor_id) as count FROM analytics_events WHERE event_type IN ('training_started', 'training_start')")
        funnel_trained = cursor.fetchone()['count']
        
        metrics['conversion_funnel'] = {
            'visitors': funnel_visitors,
            'email_provided': funnel_email,
            'trained': funnel_trained,
            'email_rate': round((funnel_email / funnel_visitors * 100) if funnel_visitors > 0 else 0, 1),
            'training_rate': round((funnel_trained / funnel_visitors * 100) if funnel_visitors > 0 else 0, 1)
        }
        
        conn.close()
        return metrics
    
    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Get feedback metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        metrics = {}
        
        # Total feedback
        cursor.execute("SELECT COUNT(*) as total FROM feedback")
        metrics['total_feedback'] = cursor.fetchone()['total']
        
        # Average rating
        cursor.execute("SELECT AVG(rating) as avg FROM feedback WHERE rating IS NOT NULL")
        result = cursor.fetchone()
        metrics['average_rating'] = round(result['avg'], 2) if result['avg'] else None
        
        # Feedback today
        cursor.execute("""
            SELECT COUNT(*) as count FROM feedback
            WHERE DATE(timestamp) = DATE('now')
        """)
        metrics['feedback_today'] = cursor.fetchone()['count']
        
        # Recent feedback messages
        cursor.execute("""
            SELECT f.category, f.rating, f.message, f.timestamp, v.email
            FROM feedback f
            LEFT JOIN visitors v ON f.visitor_id = v.id
            ORDER BY f.timestamp DESC
            LIMIT 5
        """)
        metrics['recent_feedback'] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return metrics
    
    def get_training_leaderboard(self) -> List[Dict]:
        """Get training time leaderboard by game."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                game_id,
                total_training_seconds,
                total_sessions,
                total_episodes,
                last_trained_at
            FROM game_training_stats
            ORDER BY total_training_seconds DESC
            LIMIT 10
        """)
        leaderboard = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return leaderboard
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM sessions")
        stats['total_sessions'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM episodes")
        stats['total_episodes'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM analytics_events")
        stats['total_events'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM visitors")
        stats['total_visitors'] = cursor.fetchone()['count']
        
        # Database file size
        if self.db_path.exists():
            stats['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
        else:
            stats['db_size_mb'] = 0
        
        conn.close()
        return stats


def format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def generate_report() -> str:
    """Generate the daily metrics report."""
    collector = MetricsCollector(DB_PATH)
    
    visitor_metrics = collector.get_visitor_metrics()
    session_metrics = collector.get_session_metrics()
    analytics_metrics = collector.get_analytics_metrics()
    feedback_metrics = collector.get_feedback_metrics()
    leaderboard = collector.get_training_leaderboard()
    db_stats = collector.get_database_stats()
    
    # Calculate changes
    new_visitors_change = visitor_metrics['new_visitors_today'] - visitor_metrics['new_visitors_yesterday']
    sessions_change = session_metrics['sessions_today'] - session_metrics['sessions_yesterday']
    
    # Format the report
    report = f"""
================================================================================
              ATARI RL DASHBOARD - DAILY METRICS REPORT
                     {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
================================================================================

📊 VISITOR METRICS
--------------------------------------------------------------------------------
  Total Visitors (all time):     {visitor_metrics['total_visitors']:,}
  New Visitors Today:            {visitor_metrics['new_visitors_today']:,} ({'+' if new_visitors_change >= 0 else ''}{new_visitors_change} vs yesterday)
  Returning Visitors Today:      {visitor_metrics['returning_visitors_today']:,}
  Visitors with Email:           {visitor_metrics['visitors_with_email']:,}
  Marketing Opt-ins:             {visitor_metrics['marketing_opt_ins']:,}

📈 7-DAY VISITOR TREND
--------------------------------------------------------------------------------
"""
    
    for day in visitor_metrics['visitors_trend_7d']:
        bar = '█' * min(day['count'], 50)
        report += f"  {day['date']}: {bar} {day['count']}\n"
    
    report += f"""
🎮 TRAINING SESSION METRICS
--------------------------------------------------------------------------------
  Total Sessions (all time):     {session_metrics['total_sessions']:,}
  Sessions Today:                {session_metrics['sessions_today']:,} ({'+' if sessions_change >= 0 else ''}{sessions_change} vs yesterday)
  Total Episodes Today:          {session_metrics['total_episodes_today']:,}
  Avg Session Duration:          {session_metrics['avg_session_duration_minutes']:.1f} minutes

  Sessions by Game (Today):
"""
    
    for game_id, count in session_metrics['sessions_by_game_today'].items():
        game_name = game_id.replace('ALE/', '').replace('-v5', '')
        report += f"    • {game_name}: {count}\n"
    
    if not session_metrics['sessions_by_game_today']:
        report += "    (No sessions today)\n"
    
    report += f"""
📊 CONVERSION FUNNEL (All Time)
--------------------------------------------------------------------------------
  Page Visitors:                 {analytics_metrics['conversion_funnel']['visitors']:,}
  → Email Provided:              {analytics_metrics['conversion_funnel']['email_provided']:,} ({analytics_metrics['conversion_funnel']['email_rate']:.1f}%)
  → Started Training:            {analytics_metrics['conversion_funnel']['trained']:,} ({analytics_metrics['conversion_funnel']['training_rate']:.1f}%)

📱 EVENT BREAKDOWN (Today)
--------------------------------------------------------------------------------
"""
    
    for event_type, count in list(analytics_metrics['events_today'].items())[:10]:
        report += f"  {event_type}: {count}\n"
    
    if not analytics_metrics['events_today']:
        report += "  (No events today)\n"
    
    report += f"""
🏆 TRAINING LEADERBOARD (By Total Time)
--------------------------------------------------------------------------------
"""
    
    for i, game in enumerate(leaderboard, 1):
        game_name = game['game_id'].replace('ALE/', '').replace('-v5', '')
        duration = format_duration(game['total_training_seconds'])
        report += f"  {i}. {game_name}: {duration} ({game['total_sessions']} sessions, {game['total_episodes']:,} episodes)\n"
    
    if not leaderboard:
        report += "  (No training data)\n"
    
    report += f"""
💬 FEEDBACK SUMMARY
--------------------------------------------------------------------------------
  Total Feedback:                {feedback_metrics['total_feedback']}
  Average Rating:                {feedback_metrics['average_rating'] or 'N/A'}/5
  New Feedback Today:            {feedback_metrics['feedback_today']}

  Recent Feedback:
"""
    
    for fb in feedback_metrics['recent_feedback']:
        rating_str = f"[{fb['rating']}/5]" if fb['rating'] else ""
        msg = (fb['message'][:60] + '...') if fb['message'] and len(fb['message']) > 60 else (fb['message'] or 'No message')
        report += f"    • {fb['category']} {rating_str}: {msg}\n"
    
    if not feedback_metrics['recent_feedback']:
        report += "    (No recent feedback)\n"
    
    report += f"""
💾 DATABASE STATISTICS
--------------------------------------------------------------------------------
  Total Sessions:                {db_stats['total_sessions']:,}
  Total Episodes:                {db_stats['total_episodes']:,}
  Total Analytics Events:        {db_stats['total_events']:,}
  Total Visitors:                {db_stats['total_visitors']:,}
  Database Size:                 {db_stats['db_size_mb']:.2f} MB

================================================================================
                    Generated by Atari RL Dashboard
                 https://atari.riccardocosentino.com
================================================================================
"""
    
    return report


def send_email(subject: str, body: str, to_email: str):
    """Send email report."""
    try:
        # Try using system mail command first
        process = subprocess.run(
            ['mail', '-s', subject, to_email],
            input=body.encode(),
            capture_output=True,
            timeout=30
        )
        
        if process.returncode == 0:
            print(f"Report sent to {to_email} via mail command")
            return True
        else:
            print(f"Mail command failed: {process.stderr.decode()}")
            
    except Exception as e:
        print(f"Mail command error: {e}")
    
    # Fallback to SMTP if mail command fails
    try:
        msg = MIMEMultipart()
        msg['From'] = f"Atari Dashboard <noreply@{HOSTNAME}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.send_message(msg)
        
        print(f"Report sent to {to_email} via SMTP")
        return True
        
    except Exception as e:
        print(f"SMTP error: {e}")
        return False


def main():
    """Main entry point."""
    print(f"Generating daily metrics report at {datetime.now()}")
    
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)
    
    try:
        report = generate_report()
        
        # Print to stdout
        print(report)
        
        # Send email
        subject = f"[{HOSTNAME}] Atari RL Dashboard - Daily Metrics Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        if send_email(subject, report, ADMIN_EMAIL):
            print(f"\nReport sent successfully to {ADMIN_EMAIL}")
        else:
            print(f"\nFailed to send email to {ADMIN_EMAIL}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
