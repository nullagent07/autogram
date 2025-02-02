"""
utils.py - Utility functions for AI News Agent
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class AgentUtils:
    """Utility functions for the AI News Agent"""

    @staticmethod
    def load_json(path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            if path.exists():
                return json.loads(path.read_text())
            return {}
        except Exception as e:
            print(f"Error loading JSON file {path}: {e}")
            return {}

    @staticmethod
    def save_json(data: Dict[str, Any], path: Path) -> bool:
        """Save data to JSON file with error handling"""
        try:
            path.write_text(json.dumps(data, indent=2))
            return True
        except Exception as e:
            print(f"Error saving JSON file {path}: {e}")
            return False

    @staticmethod
    def format_timestamp(dt: datetime) -> str:
        """Format datetime for logging"""
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text for file names and paths"""
        return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
        """Split text into chunks for API limits"""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


class MetricsTracker:
    """Track and analyze agent performance metrics"""

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        self.metrics = self.load_metrics()

    def load_metrics(self) -> Dict[str, Any]:
        """Load metrics from file"""
        return AgentUtils.load_json(self.metrics_file)

    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """Update metrics with new data"""
        self.metrics.update(new_metrics)
        self.save_metrics()

    def save_metrics(self) -> None:
        """Save metrics to file"""
        AgentUtils.save_json(self.metrics, self.metrics_file)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        return {
            "total_posts": self.metrics.get("successful_posts", 0),
            "success_rate": self._calculate_success_rate(),
            "average_engagement": self._calculate_average_engagement(),
            "top_topics": self._get_top_topics()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate post success rate"""
        total = self.metrics.get("successful_posts", 0) + self.metrics.get("failed_posts", 0)
        if total == 0:
            return 0.0
        return self.metrics.get("successful_posts", 0) / total * 100

    def _calculate_average_engagement(self) -> float:
        """Calculate average post engagement"""
        engagement = self.metrics.get("engagement_rates", [])
        if not engagement:
            return 0.0
        return sum(engagement) / len(engagement)

    def _get_top_topics(self, limit: int = 5) -> List[str]:
        """Get top performing topics"""
        topics = self.metrics.get("topic_performance", {})
        return sorted(topics.items(), key=lambda x: x[1], reverse=True)[:limit]
