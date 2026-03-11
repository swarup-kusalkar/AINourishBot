import json
import os
import base64
import requests
from langchain.tools import tool
from PIL import Image
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from io import BytesIO
from typing import List, Optional
import logging
from src.vector_store import (
    store_feedback, retrieve_feedback,
    store_news, retrieve_news,
    store_recipe_history, retrieve_recipe_history,
)
from src.news_scraper import fetch_health_news, process_and_store_news, get_relevant_health_tips
logging.basicConfig(level=logging.INFO)

logging.info("Extracting ingredients from image...")

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )
client = APIClient(credentials)
project_id = "skills-network"


class ExtractIngredientsTool():
    @tool("Extract ingredients")
    def extract_ingredient(image_input: str):
        """
        Extract ingredients from a food item image.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A comma-separated string of ingredients extracted from the image.
        """
        try:
            if image_input.startswith("http"):
                response = requests.get(image_input, timeout=15)
                response.raise_for_status()
                image_bytes = BytesIO(response.content)
            else:
                # Resolve to absolute path in case CWD has shifted
                abs_path = os.path.abspath(image_input)
                if not os.path.isfile(abs_path):
                    return f"ERROR: Image file not found at '{abs_path}'. Cannot extract ingredients."
                with open(abs_path, "rb") as file:
                    image_bytes = BytesIO(file.read())

            encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

            model = ModelInference(
                model_id="meta-llama/llama-3-2-90b-vision-instruct",
                credentials=credentials,
                project_id=project_id,
                params={"max_tokens": 300},
            )
            response = model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "List all visible food ingredients in this image as a comma-separated list. Only return the ingredient names, nothing else."},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                        ],
                    }
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"ExtractIngredientsTool failed: {e}")
            return f"ERROR extracting ingredients: {str(e)}. Use this error as the Final Answer and do not retry the tool."


class FilterIngredientsTool:
    @tool("Filter ingredients")
    def filter_ingredients(raw_ingredients: str) -> List[str]:
        """
        Processes the raw ingredient data and filters out non-food items or noise.
        
        :param raw_ingredients: Raw ingredients as a string.
        :return: A list of cleaned and relevant ingredients.
        """
        # Example implementation: parse the raw ingredients string into a list
        # This can be enhanced with more sophisticated parsing as needed
        ingredients = [ingredient.strip().lower() for ingredient in raw_ingredients.split(',') if ingredient.strip()]
        return ingredients

class DietaryFilterTool:
    @tool("Filter based on dietary restrictions")
    def filter_based_on_restrictions(ingredients: List[str], dietary_restrictions: Optional[str] = None) -> List[str]:
        """
        Uses an LLM model to filter ingredients based on dietary restrictions.

        :param ingredients: List of ingredients.
        :param dietary_restrictions: Dietary restrictions (e.g., vegan, gluten-free). Defaults to None.
        :return: Filtered list of ingredients that comply with the dietary restrictions.
        """
        # If no dietary restrictions are provided, return the original ingredients
        if not dietary_restrictions:
            return ingredients

        # Initialize the WatsonX model
        model = ModelInference(
            model_id="ibm/granite-4-h-small",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 150},
        )

        # Create a prompt for the LLM to filter ingredients
        prompt = f"""
        You are an AI nutritionist specialized in dietary restrictions. 
        Given the following list of ingredients: {', '.join(ingredients)}, 
        and the dietary restriction: {dietary_restrictions}, 
        remove any ingredient that does not comply with this restriction. 
        Return only the compliant ingredients as a comma-separated list with no additional commentary.
        """

        # Send the prompt to the model for filtering
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
        )

        # Parse the response to return the filtered list
        filtered = response['choices'][0]['message']['content'].strip().lower()
        filtered_list = [item.strip() for item in filtered.split(',') if item.strip()]
        return filtered_list

    
class NutrientAnalysisTool():
    @tool("Analyze nutritional values and calories of the dish from uploaded image")
    def analyze_image(image_input: str):
        """
        Provide a detailed nutrient breakdown and estimate the total calories of all ingredients from the uploaded image.
        
        :param image_input: The image file path (local) or URL (remote).
        :return: A string with nutrient breakdown (protein, carbs, fat, etc.) and estimated calorie information.
        """
        if image_input.startswith("http"):  # Check if input is a URL
            # Download the image from the URL
            response = requests.get(image_input)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
        else:
            # Open the local image file in binary mode
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"No file found at path: {image_input}")
            with open(image_input, "rb") as file:
                image_bytes = BytesIO(file.read())

        # Encode the image to a base64 string
        encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

        # Call the model with the encoded image
        model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 300},
        )
        # Assistant prompt (can be customized)
        assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:
        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories
        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories
        3. **Total Calories**: Provide the total number of calories for all food items.
        Example:
        Total Calories: [Number of Calories]
        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.
        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]
        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.
        6. **Disclaimer**: Include the following exact text as a disclaimer:
        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.
        Format your response exactly like the template above to ensure consistency.
        """
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": assistant_prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                    ],
                }
            ]
        )

        return response['choices'][0]['message']['content']


class FeedbackTool:
    @tool("Store user feedback")
    def store_user_feedback(feedback_text: str) -> str:
        """
        Store user feedback about a recipe into the vector database.
        Input should be a JSON string with keys: recipe_title, rating (1-5), comment, dietary_restrictions (optional).
        
        :param feedback_text: JSON string with feedback details.
        :return: Confirmation message.
        """
        data = json.loads(feedback_text)
        return store_feedback(
            recipe_title=data.get("recipe_title", ""),
            rating=int(data.get("rating", 3)),
            comment=data.get("comment", ""),
            dietary_restrictions=data.get("dietary_restrictions", ""),
        )

    @tool("Retrieve past feedback")
    def get_past_feedback(query: str) -> str:
        """
        Retrieve past user feedback relevant to the current context from the vector database.
        
        :param query: A search query describing the context (e.g., ingredients, dietary restrictions).
        :return: JSON string of relevant past feedback entries, or a message indicating no feedback exists.
        """
        results = retrieve_feedback(query=query, n_results=5)
        if not results:
            return json.dumps({
                "status": "no_data",
                "message": "No past feedback found in the database. This appears to be a first-time session. Proceed to generate recipe suggestions based solely on the filtered ingredients without any personalization from past feedback.",
                "feedback": []
            })
        return json.dumps({"status": "found", "feedback": results}, default=str)


class HealthNewsTool:
    @tool("Fetch and store health news")
    def fetch_and_store_news(query: str) -> str:
        """
        Fetch latest health and nutrition news from trusted RSS feeds and store them in the vector database.
        
        :param query: Not used for fetching, but triggers the fetch. Pass any string.
        :return: JSON string of fetched and stored news articles.
        """
        articles = fetch_health_news(max_per_feed=3)
        stored = process_and_store_news(articles)
        return json.dumps(stored, default=str)

    @tool("Get relevant health tips")
    def get_health_tips(query: str) -> str:
        """
        Retrieve health tips from the vector database that are relevant to the given ingredients or dietary context.
        
        :param query: The ingredients or dietary context to search for relevant health tips.
        :return: JSON string of relevant health news and tips, or a message indicating no tips exist yet.
        """
        tips = get_relevant_health_tips(query=query, n_results=5)
        if not tips:
            return json.dumps({
                "status": "no_data",
                "message": "No health news tips are stored in the database yet. Proceed to generate recipe suggestions based solely on the filtered ingredients using general nutritional knowledge. Do not wait for health news data.",
                "tips": []
            })
        return json.dumps({"status": "found", "tips": tips}, default=str)


class RecipeMemoryTool:
    @tool("Store recipe to history")
    def store_recipe(recipe_json: str) -> str:
        """
        Store a generated recipe into the recipe history vector database for future learning.
        Input should be a JSON string with keys: recipe_title, ingredients (list), health_score (1-10), prep_time, dietary_restrictions.

        :param recipe_json: JSON string with recipe details.
        :return: Confirmation message.
        """
        data = json.loads(recipe_json)
        return store_recipe_history(
            recipe_title=data.get("recipe_title", ""),
            ingredients=data.get("ingredients", []),
            health_score=int(data.get("health_score", 5)),
            prep_time=data.get("prep_time", "unknown"),
            dietary_restrictions=data.get("dietary_restrictions", ""),
        )

    @tool("Retrieve recipe history")
    def get_recipe_history(query: str) -> str:
        """
        Retrieve past recipe history from the vector database to inform current recipe suggestions.
        
        :param query: The context to search for (e.g., ingredients, dietary preferences).
        :return: JSON string of relevant past recipes, or a message indicating no history exists.
        """
        results = retrieve_recipe_history(query=query, n_results=5)
        if not results:
            return json.dumps({
                "status": "no_data",
                "message": "No recipe history found in the database. Proceed to generate new recipe suggestions based on the filtered ingredients without any historical recipe context.",
                "history": []
            })
        return json.dumps({"status": "found", "history": results}, default=str)
