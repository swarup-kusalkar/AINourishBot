from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Recipe(BaseModel):
    title: str = Field(..., description="Recipe title")
    ingredients: List[str] = Field(..., description="List of ingredients required for the recipe")
    instructions: str = Field(..., description="Step-by-step cooking instructions")
    calorie_estimate: int = Field(..., description="Estimated calories per serving")
    health_score: Optional[int] = Field(None, description="Health score out of 10")
    prep_time: Optional[str] = Field(None, description="Estimated preparation time")

class QuickMeal(BaseModel):
    title: str = Field(..., description="Quick meal title")
    ingredients: List[str] = Field(..., description="List of ingredients")
    instructions: str = Field(..., description="Quick step-by-step instructions")
    calorie_estimate: int = Field(..., description="Estimated calories per serving")
    prep_time: str = Field(..., description="Preparation time (should be under 15 minutes)")
    health_score: int = Field(..., description="Health score out of 10")
    why_healthy: str = Field(..., description="Brief explanation of why this is the healthiest quick option")

class HealthNewsTip(BaseModel):
    headline: str = Field(..., description="News headline")
    summary: str = Field(..., description="Brief summary of the health news")
    health_tip: str = Field(..., description="Actionable health tip extracted from the news")
    source: str = Field(..., description="Source of the news")

class RecipeSuggestionOutput(BaseModel):
    recipes: List[Recipe] = Field(..., description="List of suggested recipes")
    quick_meal: Optional[QuickMeal] = Field(None, description="Best quick healthy meal suggestion")
    health_news_tips: Optional[List[HealthNewsTip]] = Field(default=None, description="Relevant health news tips that influenced the recipes")
    personalization_note: Optional[str] = Field(None, description="Note about how past feedback influenced these suggestions")

class VitaminInfo(BaseModel):
    name: str = Field(..., description="Name of the vitamin")
    percentage_dv: str = Field(..., description="Percentage of the Daily Value")

class MineralInfo(BaseModel):
    name: str = Field(..., description="Name of the mineral")
    amount: str = Field(..., description="Amount and unit of the mineral")

class NutrientBreakdown(BaseModel):
    protein: Optional[str] = Field(None, description="Protein content")
    carbohydrates: Optional[str] = Field(None, description="Carbohydrates content")
    fats: Optional[str] = Field(None, description="Fats content")
    vitamins: List[VitaminInfo] = Field(default_factory=list, description="List of vitamins and their %DV")
    minerals: List[MineralInfo] = Field(default_factory=list, description="List of minerals and their amounts")

class NutrientAnalysisOutput(BaseModel):
    dish: Optional[str] = Field(None, description="Identified dish")
    portion_size: Optional[str] = Field(None, description="Portion size description")
    estimated_calories: Optional[int] = Field(None, description="Estimated calories per portion")
    nutrients: NutrientBreakdown = Field(default_factory=NutrientBreakdown, description="Detailed nutrient breakdown")
    health_evaluation: Optional[str] = Field(None, description="Health evaluation summary")

