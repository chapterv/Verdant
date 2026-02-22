"""
WebArena Benchmark Tasks

Based on the WebArena benchmark for Web Agent evaluation.
Reference: https://webarena.dev/

This module provides task definitions for WebArena-style evaluations.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskCategory(Enum):
    """WebArena Task Categories"""
    ECOMMERCE = "e-commerce"
    SOCIAL_FORUM = "social-forum"
    REDDIT = "reddit"
    GITHUB = "github"
    CHATGROUND = "chatground"
    CMS = "cms"


class TaskDifficulty(Enum):
    """Task Difficulty Levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class WebArenaTask:
    """WebArena Task Definition"""
    id: str
    category: TaskCategory
    difficulty: TaskDifficulty
    description: str
    instruction: str
    start_url: str
    expected_content: List[str] = field(default_factory=list)
    forbidden_content: List[str] = field(default_factory=list)
    max_steps: int = 20
    timeout: int = 300
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)


# WebArena Task Definitions
WEBARENA_TASKS = [
    # E-commerce Tasks
    WebArenaTask(
        id="webarena_ecom_001",
        category=TaskCategory.ECOMMERCE,
        difficulty=TaskDifficulty.EASY,
        description="Find a product and add it to cart",
        instruction="Navigate to the shopping site, find a specific product (e.g., a laptop), and add it to the shopping cart. Verify the item appears in the cart.",
        start_url="https://shopping.example.com",
        expected_content=["added to cart", "shopping cart", "laptop"],
        max_steps=10,
        evaluation_criteria={
            "cart_has_item": True,
            "correct_product": True
        }
    ),
    WebArenaTask(
        id="webarena_ecom_002",
        category=TaskCategory.ECOMMERCE,
        difficulty=TaskDifficulty.MEDIUM,
        description="Complete a purchase flow",
        instruction="Find a product, add it to cart, proceed to checkout, fill in shipping information, and complete the purchase. Do not actually process payment.",
        start_url="https://shopping.example.com",
        expected_content=["order confirmation", "thank you", "order placed"],
        max_steps=20,
        evaluation_criteria={
            "checkout_completed": True,
            "order_created": True
        }
    ),
    WebArenaTask(
        id="webarena_ecom_003",
        category=TaskCategory.ECOMMERCE,
        difficulty=TaskDifficulty.HARD,
        description="Compare products and write review",
        instruction="Find at least 3 similar products, compare their prices and ratings, write a review for the best product, and post it.",
        start_url="https://shopping.example.com/categories/laptops",
        expected_content=["review posted", "review submitted"],
        max_steps=30,
        evaluation_criteria={
            "comparison_done": True,
            "review_posted": True
        }
    ),
    
    # Social Forum Tasks
    WebArenaTask(
        id="webarena_forum_001",
        category=TaskCategory.SOCIAL_FORUM,
        difficulty=TaskDifficulty.EASY,
        description="Find and read a forum post",
        instruction="Navigate to the forum, find discussions about a specific topic, and read the most popular post.",
        start_url="https://forum.example.com",
        expected_content=["post content", "comments"],
        max_steps=8,
        evaluation_criteria={
            "post_found": True,
            "post_read": True
        }
    ),
    WebArenaTask(
        id="webarena_forum_002",
        category=TaskCategory.SOCIAL_FORUM,
        difficulty=TaskDifficulty.MEDIUM,
        description="Create a forum post",
        instruction="Register an account on the forum, create a new post with a specific topic, and add content to it.",
        start_url="https://forum.example.com",
        expected_content=["post created", "new discussion"],
        max_steps=15,
        evaluation_criteria={
            "account_created": True,
            "post_created": True
        }
    ),
    WebArenaTask(
        id="webarena_forum_003",
        category=TaskCategory.SOCIAL_FORUM,
        difficulty=TaskDifficulty.HARD,
        description="Engage in a discussion thread",
        instruction="Find a controversial discussion, add a thoughtful reply that contributes to the conversation, and engage with at least one other user's response.",
        start_url="https://forum.example.com",
        expected_content=["reply posted", "response added"],
        max_steps=25,
        evaluation_criteria={
            "reply_posted": True,
            "engagement": True
        }
    ),
    
    # GitHub Tasks
    WebArenaTask(
        id="webarena_github_001",
        category=TaskCategory.GITHUB,
        difficulty=TaskDifficulty.EASY,
        description="Find a repository and star it",
        instruction="Navigate to GitHub, search for a repository related to machine learning, and star it.",
        start_url="https://github.com",
        expected_content=["starred", "repository"],
        max_steps=8,
        evaluation_criteria={
            "repo_found": True,
            "repo_starred": True
        }
    ),
    WebArenaTask(
        id="webarena_github_002",
        category=TaskCategory.GITHUB,
        difficulty=TaskDifficulty.MEDIUM,
        description="Create an issue on a repository",
        instruction="Find a popular open-source repository, create a new issue reporting a bug with a clear description and steps to reproduce.",
        start_url="https://github.com",
        expected_content=["issue created", "new issue"],
        max_steps=15,
        evaluation_criteria={
            "issue_created": True,
            "issue_has_details": True
        }
    ),
    WebArenaTask(
        id="webarena_github_003",
        category=TaskCategory.GITHUB,
        difficulty=TaskDifficulty.HARD,
        description="Submit a pull request",
        instruction="Fork a repository, make a small code change (e.g., fix a typo in README), and submit a pull request with a clear description.",
        start_url="https://github.com",
        expected_content=["pull request", "PR created"],
        max_steps=30,
        evaluation_criteria={
            "fork_created": True,
            "pr_submitted": True
        }
    ),
    
    # Reddit-style Tasks
    WebArenaTask(
        id="webarena_reddit_001",
        category=TaskCategory.REDDIT,
        difficulty=TaskDifficulty.EASY,
        description="Find and upvote a post",
        instruction="Navigate to Reddit, find a post in a specific subreddit about technology, and upvote it.",
        start_url="https://reddit.com",
        expected_content=["upvoted", "vote recorded"],
        max_steps=8,
        evaluation_criteria={
            "post_found": True,
            "upvoted": True
        }
    ),
    WebArenaTask(
        id="webarena_reddit_002",
        category=TaskCategory.REDDIT,
        difficulty=TaskDifficulty.MEDIUM,
        description="Create a post in a subreddit",
        instruction="Create a new post in a specific subreddit with a title and body content. Include at least one image or link.",
        start_url="https://reddit.com/r/technology",
        expected_content=["post submitted", "your post"],
        max_steps=15,
        evaluation_criteria={
            "post_created": True,
            "post_has_content": True
        }
    ),
    
    # CMS Tasks
    WebArenaTask(
        id="webarena_cms_001",
        category=TaskCategory.CMS,
        difficulty=TaskDifficulty.EASY,
        description="Navigate CMS and find content",
        instruction="Log into the CMS, navigate to the content management section, and find a specific article.",
        start_url="https://cms.example.com/admin",
        expected_content=["article found", "content"],
        max_steps=10,
        evaluation_criteria={
            "logged_in": True,
            "article_found": True
        }
    ),
    WebArenaTask(
        id="webarena_cms_002",
        category=TaskCategory.CMS,
        difficulty=TaskDifficulty.MEDIUM,
        description="Edit and publish content",
        instruction="Log into the CMS, create a new article with title and body, save it as draft, preview it, and then publish.",
        start_url="https://cms.example.com/admin",
        expected_content=["article published", "published"],
        max_steps=20,
        evaluation_criteria={
            "article_created": True,
            "article_published": True
        }
    ),
]


def get_tasks_by_category(category: TaskCategory) -> List[WebArenaTask]:
    """Get all tasks for a specific category"""
    return [t for t in WEBARENA_TASKS if t.category == category]


def get_tasks_by_difficulty(difficulty: TaskDifficulty) -> List[WebArenaTask]:
    """Get all tasks for a specific difficulty"""
    return [t for t in WEBARENA_TASKS if t.difficulty == difficulty]


def get_task_by_id(task_id: str) -> Optional[WebArenaTask]:
    """Get a specific task by ID"""
    for task in WEBARENA_TASKS:
        if task.id == task_id:
            return task
    return None


def get_all_task_ids() -> List[str]:
    """Get all task IDs"""
    return [t.id for t in WEBARENA_TASKS]


# Convert to internal Task format for Green Agent
def to_internal_task(webarena_task: WebArenaTask) -> Dict[str, Any]:
    """Convert WebArena task to internal Task format"""
    from .agent import Task
    
    return Task(
        id=webarena_task.id,
        description=webarena_task.description,
        target_url=webarena_task.start_url,
        expected_result=", ".join(webarena_task.expected_content),
        timeout=webarena_task.timeout,
        metadata={
            "category": webarena_task.category.value,
            "difficulty": webarena_task.difficulty.value,
            "instruction": webarena_task.instruction,
            "evaluation_criteria": webarena_task.evaluation_criteria,
            "expected_content": webarena_task.expected_content,
            "forbidden_content": webarena_task.forbidden_content,
            "max_steps": webarena_task.max_steps,
            "benchmark": "webarena"
        }
    )


class TaskSet:
    """Collection of tasks for evaluation"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tasks: List[WebArenaTask] = []
    
    def add_task(self, task: WebArenaTask):
        """Add a task to the task set"""
        self.tasks.append(task)
    
    def add_tasks_by_category(self, category: TaskCategory):
        """Add all tasks from a category"""
        self.tasks.extend(get_tasks_by_category(category))
    
    def add_tasks_by_difficulty(self, difficulty: TaskDifficulty):
        """Add all tasks of a specific difficulty"""
        self.tasks.extend(get_tasks_by_difficulty(difficulty))
    
    def to_internal_tasks(self) -> List[Dict[str, Any]]:
        """Convert all tasks to internal format"""
        return [to_internal_task(t) for t in self.tasks]
    
    def __len__(self):
        return len(self.tasks)


# Predefined Task Sets
def create_full_webarena_task_set() -> TaskSet:
    """Create a task set with all WebArena tasks"""
    task_set = TaskSet(
        name="WebArena Full",
        description="Complete WebArena benchmark with all categories"
    )
    task_set.tasks = WEBARENA_TASKS.copy()
    return task_set


def create_easy_task_set() -> TaskSet:
    """Create a task set with only easy tasks"""
    task_set = TaskSet(
        name="WebArena Easy",
        description="Easy WebArena tasks for initial evaluation"
    )
    task_set.add_tasks_by_difficulty(TaskDifficulty.EASY)
    return task_set


def create_ecommerce_task_set() -> TaskSet:
    """Create a task set with e-commerce tasks"""
    task_set = TaskSet(
        name="WebArena E-commerce",
        description="E-commerce specific tasks"
    )
    task_set.add_tasks_by_category(TaskCategory.ECOMMERCE)
    return task_set
