"""
AI News Agent
An autonomous agent for AI news curation and social media management.

Author: [Your Name]
Version: 2.0.0
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import aiohttp
import google.generativeai as genai
import replicate
import requests
from dotenv import load_dotenv
from gnews import GNews
from instagrapi import Client
from pydantic import BaseModel

load_dotenv()


class AgentState(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting_news"
    ANALYZING = "analyzing_content"
    GENERATING = "generating_content"
    POSTING = "posting_content"
    ERROR = "error"


class AgentMetrics(BaseModel):
    news_collected: int = 0
    posts_created: int = 0
    successful_posts: int = 0
    failed_posts: int = 0
    last_run: Optional[datetime] = None
    average_engagement: float = 0.0
    top_performing_topics: List[str] = []


@dataclass
class AgentMemory:
    posted_titles: Set[str] = None
    performance_history: Dict[str, float] = None
    topic_performance: Dict[str, float] = None

    def __post_init__(self):
        self.posted_titles = set()
        self.performance_history = {}
        self.topic_performance = {}

    def save(self, path: Path) -> None:
        data = {
            "posted_titles": list(self.posted_titles),
            "performance_history": self.performance_history,
            "topic_performance": self.topic_performance
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> 'AgentMemory':
        if not path.exists():
            return cls()

        data = json.loads(path.read_text())
        memory = cls()
        memory.posted_titles = set(data["posted_titles"])
        memory.performance_history = data["performance_history"]
        memory.topic_performance = data["topic_performance"]
        return memory


class AgentConfig(BaseModel):
    name: str = "AI News Agent"
    personality: str = "professional"
    news_sources: List[str] = ["gnews"]
    post_frequency: int = 3  # posts per day
    max_news_age: int = 24  # hours
    engagement_threshold: float = 0.5
    memory_path: Path = Path("../agent_memory.json")
    output_dir: Path = Path("../output")
    credentials: Dict[str, str] = {}


class AINewsAgent:

    def __init__(self, config: AgentConfig):
        self.config = config
        self._state = AgentState.IDLE
        self.memory = AgentMemory.load(config.memory_path)
        self.metrics = AgentMetrics()
        self.logger = self._setup_logging()

        self._init_ai_components()

        self.logger.info(f"Agent '{config.name}' initialized")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: AgentState):
        """Log state changes whenever the state updates."""
        if self._state != new_state:
            self.logger.info(f"{new_state.value.capitalize().replace('_', ' ')}...")
        self._state = new_state

    def _setup_logging(self) -> logging.Logger:
        """Set up agent logging"""
        logger = logging.getLogger(self.config.name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        fh = logging.FileHandler(f"{self.config.name.lower()}_agent.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def _init_ai_components(self) -> None:
        try:
            if self.config.credentials.get("GEMINI_API_KEY"):
                genai.configure(api_key=self.config.credentials["GEMINI_API_KEY"])
                self.content_analyzer = genai.GenerativeModel("gemini-1.5-flash")
            else:
                self.content_analyzer = None
                self.logger.warning("Gemini AI not configured")

            if self.config.credentials.get("REPLICATE_API_TOKEN"):
                self.image_generator = replicate.Client(
                    api_token=self.config.credentials["REPLICATE_API_TOKEN"]
                )
            else:
                self.image_generator = None
                self.logger.warning("Image generator not configured")

            if all(k in self.config.credentials for k in ["IG_USERNAME", "IG_PASSWORD"]):
                self.instagram = Client()
                self._login_instagram()
            else:
                self.instagram = None
                self.logger.warning("Instagram client not configured")

        except Exception as e:
            self.logger.error(f"Failed to initialize AI components: {e}")
            self.state = AgentState.ERROR

    def _login_instagram(self) -> None:
        """Login to Instagram with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.instagram.login(
                    self.config.credentials["IG_USERNAME"],
                    self.config.credentials["IG_PASSWORD"]
                )
                self.logger.info("Successfully logged into Instagram")
                return
            except Exception as e:
                self.logger.warning(f"Login attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("Instagram login failed after max retries")
                    raise

    async def run(self) -> None:
        """Main agent loop"""
        self.logger.info("Agent starting...")

        while True:
            try:
                await self._execute_cycle()
                self.memory.save(self.config.memory_path)
                self.metrics.last_run = datetime.now()
                await asyncio.sleep(self._calculate_next_run_delay())

            except Exception as e:
                self.logger.error(f"Error in agent cycle: {e}")
                self.state = AgentState.ERROR
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _execute_cycle(self) -> None:
        """Execute one full agent cycle"""
        self.state = AgentState.COLLECTING
        news_items = await self._collect_news()

        if not news_items:
            self.logger.info("No new news items found")
            return

        self.state = AgentState.ANALYZING
        selected_news = await self._analyze_news(news_items)

        if not selected_news:
            self.logger.info("No suitable news items for posting")
            return

        self.state = AgentState.GENERATING
        content = await self._generate_content(selected_news)

        if not content:
            self.logger.warning("Failed to generate content")
            return

        self.state = AgentState.POSTING
        success = await self._post_content(content)

        if success:
            self._update_memory(selected_news, content)
            self.metrics.successful_posts += 1
        else:
            self.metrics.failed_posts += 1

        self.state = AgentState.IDLE

    async def _collect_news(self) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.config.news_sources:
                tasks.append(self._fetch_from_source(session, source))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_news = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching news: {result}")
                    continue
                all_news.extend(result)

            self.metrics.news_collected += len(all_news)
            return all_news

    async def _fetch_from_source(self, session: aiohttp.ClientSession, source: str) -> List[Dict[str, Any]]:
        if source == "gnews":
            return await self._fetch_gnews()
        # TODO: Add more sources here
        return []

    async def _fetch_gnews(self) -> List[Dict[str, Any]]:
        """Fetch news from GNews"""
        try:
            gnews = GNews(max_results=10, period='1d', country='US')
            news = gnews.get_news('AI News')
            return news
        except Exception as e:
            self.logger.error(f"Error fetching from GNews: {e}")
            return []

    async def _analyze_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.content_analyzer:
            return news_items[:3]  # Default selection if no AI available

        try:
            # Prepare news for analysis
            titles = [item['title'] for item in news_items]
            titles_str = "\n".join(titles)

            prompt = (
                f"Analyze these AI news titles and select the top 3 most important "
                f"and engaging ones for social media. Consider:\n"
                f"1. News importance and impact\n"
                f"2. Social media engagement potential\n"
                f"3. Current AI trends and interests\n\n"
                f"Titles:\n{titles_str}\n\n"
                f"Return only the selected titles, one per line."
            )

            response = await asyncio.to_thread(
                self.content_analyzer.generate_content,
                prompt
            )

            selected_titles = response.text.strip().split("\n")
            return [
                item for item in news_items
                if item['title'] in selected_titles
            ]

        except Exception as e:
            self.logger.error(f"News analysis failed: {e}")
            return news_items[:3]

    async def _generate_content(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate content for posting"""
        try:
            content = {
                "images": [],
                "caption": "",
                "hashtags": []
            }

            for item in news_items:
                image = await self._generate_image(item['title'])
                if image:
                    content["images"].append(image)
                content["caption"] += f"{item['title']}\n\n"

            content["hashtags"] = await self._generate_hashtags()

            content["caption"] += "\n" + " ".join(content["hashtags"])

            return content

        except Exception as e:
            self.logger.error(f"Content generation failed: {e}")
            return None

    async def _generate_image(self, text: str) -> Optional[Path]:
        if not self.image_generator:
            return None

        try:
            # Enhanced prompt for better image generation
            prompt = await self._enhance_prompt(text)

            output = await asyncio.to_thread(
                self.image_generator.run,
                "ideogram-ai/ideogram-v2",
                {
                    "prompt": prompt,
                    "negative_prompt": "text, watermark, low quality",
                    "width": 1024,
                    "height": 1024,
                }
            )

            image_path =  f"output/{text[:50].replace(' ', '_')}.png"
            async with aiohttp.ClientSession() as session:
                async with session.get(output.url) as response:
                    image_data = await response.read()
                    Path(image_path).write_bytes(image_data)

            return image_path

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return None

    async def _enhance_prompt(self, text: str) -> str:
        """Enhance text prompt for better image generation"""
        if not self.content_analyzer:
            return f"Professional tech visualization about: {text}"

        try:
            prompt = (
                f"Convert this news title into an engaging visual prompt:\n{text}\n"
                f"Make it suitable for a professional tech-focused social media post."
            )

            response = await asyncio.to_thread(
                self.content_analyzer.generate_content,
                prompt
            )

            return response.text.strip()

        except Exception as e:
            self.logger.error(f"Prompt enhancement failed: {e}")
            return f"Modern tech visualization: {text}"

    async def _generate_hashtags(self) -> List[str]:
        """Generate relevant hashtags"""
        default_tags = ["#AI", "#ArtificialIntelligence", "#Tech", "#Innovation"]

        if not self.content_analyzer:
            return default_tags

        try:
            prompt = (
                "Generate 8 relevant hashtags for an AI news post. "
                "Include trending tech hashtags. Return only the hashtags, "
                "one per line, without numbers or explanations."
            )

            response = await asyncio.to_thread(
                self.content_analyzer.generate_content,
                prompt
            )

            hashtags = response.text.strip().split("\n")
            return [tag if tag.startswith("#") else f"#{tag}" for tag in hashtags]

        except Exception as e:
            self.logger.error(f"Hashtag generation failed: {e}")
            return default_tags

    async def _post_content(self, content: Dict[str, Any]) -> bool:
        if not self.instagram:
            self.logger.warning("Instagram client not configured")
            return False

        images = content.get("images", [])
        caption = content.get("caption", "")

        # Validate images
        valid_images = []
        for img in images:
            path = Path(img)
            if not path.exists():
                self.logger.error(f"Missing image: {img}")
                continue
            valid_images.append(str(path.resolve()))

        if not valid_images:
            self.logger.error("No valid images to post")
            return False

        try:
            if len(valid_images) == 1:
                result = await asyncio.to_thread(
                    self.instagram.photo_upload,
                    valid_images[0],
                    caption
                )
            else:
                result = await asyncio.to_thread(
                    self.instagram.album_upload,
                    valid_images,
                    caption
                )

            self.logger.info(f"Posted successfully! Media ID: {result.id}")
            return True

        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            return False

    def _update_memory(self, news_items: List[Dict[str, Any]], content: Dict[str, Any]) -> None:
        """Update agent's memory with new content"""
        for item in news_items:
            self.memory.posted_titles.add(item['title'])

        # Save memory
        self.memory.save(self.config.memory_path)

    def _calculate_next_run_delay(self) -> int:
        """Calculate delay until next run based on post frequency"""
        seconds_per_day = 86400
        delay = seconds_per_day / self.config.post_frequency
        return int(delay)
