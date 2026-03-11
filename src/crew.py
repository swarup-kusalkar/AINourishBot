import os
import yaml
import base64
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.tools import (
    ExtractIngredientsTool, 
    FilterIngredientsTool, 
    DietaryFilterTool,
    NutrientAnalysisTool,
    FeedbackTool,
    HealthNewsTool,
    RecipeMemoryTool,
)
from ibm_watsonx_ai import Credentials, APIClient
from src.models import RecipeSuggestionOutput, NutrientAnalysisOutput 

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )
client = APIClient(credentials)
project_id = "skills-network"

# Get the absolute path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

@CrewBase
class BaseNourishBotCrew:
    agents_config_path = os.path.join(CONFIG_DIR, 'agents.yaml')
    tasks_config_path = os.path.join(CONFIG_DIR, 'tasks.yaml')
    
    def __init__(self, image_data, dietary_restrictions: str = None):
        self.image_data = image_data
        self.dietary_restrictions = dietary_restrictions

        with open(self.agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        
        with open(self.tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

    @agent
    def ingredient_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['ingredient_detection_agent'],
            tools=[
                ExtractIngredientsTool.extract_ingredient, 
                FilterIngredientsTool.filter_ingredients
            ],
            allow_delegation=False,
            max_iter=5,
            verbose=True
        )

    @agent
    def dietary_filtering_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['dietary_filtering_agent'],
            tools=[DietaryFilterTool.filter_based_on_restrictions],
            allow_delegation=True,
            max_iter=6,
            verbose=True
        )

    @agent
    def nutrient_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['nutrient_analysis_agent'],
            tools=[NutrientAnalysisTool.analyze_image],
            allow_delegation=False,
            max_iter=4,
            verbose=True
        )

    @agent
    def recipe_suggestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['recipe_suggestion_agent'],
            tools=[
                RecipeMemoryTool.store_recipe,
                RecipeMemoryTool.get_recipe_history,
                FeedbackTool.get_past_feedback,
                HealthNewsTool.get_health_tips,
            ],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def feedback_learning_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['feedback_learning_agent'],
            tools=[
                FeedbackTool.get_past_feedback,
                FeedbackTool.store_user_feedback,
            ],
            allow_delegation=False,
            max_iter=4,
            verbose=True
        )

    @agent
    def news_curator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['news_curator_agent'],
            tools=[
                HealthNewsTool.fetch_and_store_news,
                HealthNewsTool.get_health_tips,
            ],
            allow_delegation=False,
            max_iter=4,
            verbose=True
        )

    @agent
    def quick_meal_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['quick_meal_agent'],
            tools=[
                RecipeMemoryTool.get_recipe_history,
                HealthNewsTool.get_health_tips,
                FeedbackTool.get_past_feedback,
            ],
            allow_delegation=False,
            max_iter=4,
            verbose=True
        )

    @task
    def ingredient_detection_task(self) -> Task:
        task_config = self.tasks_config['ingredient_detection_task']

        return Task(
            description=task_config['description'],
            agent=self.ingredient_detection_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def dietary_filtering_task(self) -> Task:
        task_config = self.tasks_config['dietary_filtering_task']

        return Task(
            description=task_config['description'],
            agent=self.dietary_filtering_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def nutrient_analysis_task(self) -> Task:
        task_config = self.tasks_config['nutrient_analysis_task']

        return Task(
            description=task_config['description'],
            agent=self.nutrient_analysis_agent(),
            expected_output=task_config['expected_output'],
            output_json=NutrientAnalysisOutput
        )

    @task
    def recipe_suggestion_task(self) -> Task:
        task_config = self.tasks_config['recipe_suggestion_task']

        return Task(
            description=task_config['description'],
            agent=self.recipe_suggestion_agent(),
            expected_output=task_config['expected_output'],
            output_json=RecipeSuggestionOutput
        )

    @task
    def feedback_retrieval_task(self) -> Task:
        task_config = self.tasks_config['feedback_retrieval_task']

        return Task(
            description=task_config['description'],
            agent=self.feedback_learning_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def news_fetch_task(self) -> Task:
        task_config = self.tasks_config['news_fetch_task']

        return Task(
            description=task_config['description'],
            agent=self.news_curator_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def news_retrieval_task(self) -> Task:
        task_config = self.tasks_config['news_retrieval_task']

        return Task(
            description=task_config['description'],
            agent=self.news_curator_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def quick_meal_task(self) -> Task:
        task_config = self.tasks_config['quick_meal_task']

        return Task(
            description=task_config['description'],
            agent=self.quick_meal_agent(),
            expected_output=task_config['expected_output']
        )


@CrewBase
class NourishBotRecipeCrew(BaseNourishBotCrew):

    @crew
    def crew(self) -> Crew:
        # Create task instances so we can reference them for explicit context chaining
        ingredient_task = self.ingredient_detection_task()
        dietary_task = self.dietary_filtering_task()
        feedback_task = self.feedback_retrieval_task()
        news_task = self.news_retrieval_task()

        # Build recipe task with explicit context so the agent receives the filtered
        # ingredients directly, not just the immediately-preceding task's output.
        # RecipeSuggestionOutput already contains a quick_meal field, so this single
        # task produces both recipes and the quick-meal suggestion as structured JSON.
        recipe_task = Task(
            description=self.tasks_config['recipe_suggestion_task']['description'],
            agent=self.recipe_suggestion_agent(),
            context=[dietary_task, feedback_task, news_task],
            expected_output=self.tasks_config['recipe_suggestion_task']['expected_output'],
            output_json=RecipeSuggestionOutput
        )

        tasks = [ingredient_task, dietary_task, feedback_task, news_task, recipe_task]
        agents = [
            self.ingredient_detection_agent(),
            self.dietary_filtering_agent(),
            self.feedback_learning_agent(),
            self.news_curator_agent(),
            self.recipe_suggestion_agent(),
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )


@CrewBase
class NourishBotAnalysisCrew(BaseNourishBotCrew):

    @crew
    def crew(self) -> Crew:
        tasks = [
            self.nutrient_analysis_task(),
        ]

        agents = [
            self.nutrient_analysis_agent(),
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )


@CrewBase
class NourishBotNewsCrew(BaseNourishBotCrew):
    """Crew dedicated to fetching and storing health news."""

    def __init__(self):
        super().__init__(image_data=None, dietary_restrictions=None)

    @crew
    def crew(self) -> Crew:
        tasks = [
            self.news_fetch_task(),
        ]

        agents = [
            self.news_curator_agent(),
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
