import gradio as gr
import json
import base64
import time
import os
from src.crew import NourishBotRecipeCrew, NourishBotAnalysisCrew, NourishBotNewsCrew
from src.vector_store import store_feedback
from src.news_scraper import fetch_health_news, process_and_store_news, get_relevant_health_tips

# Module-level state to hold the last generated recipes for feedback
_last_recipes = []

def format_recipe_output(final_output):
    """
    Formats the recipe output into a table-based Markdown format,
    including quick meal, health news tips, and personalization notes.
    
    :param final_output: The output from the NourishBotRecipe workflow.
    :return: Formatted output as a Markdown string.
    """
    global _last_recipes
    output = "## 🍽 Recipe Ideas\n\n"
    recipes = []

    # Check if final_output directly contains recipes
    if "recipes" in final_output:
        recipes = final_output["recipes"]
    else:
        # Fallback: try to extract from nested task output
        recipe_task_output = final_output.get("recipe_suggestion_task")
        if recipe_task_output and hasattr(recipe_task_output, "json_dict") and recipe_task_output.json_dict:
            recipes = recipe_task_output.json_dict.get("recipes", [])
    
    # Store recipe titles for feedback dropdown
    _last_recipes = [r.get("title", f"Recipe {i+1}") for i, r in enumerate(recipes)]

    # Show personalization note if available
    if note := final_output.get("personalization_note"):
        output += f"> 🧠 **Personalized for you:** {note}\n\n"

    if recipes:
        for idx, recipe in enumerate(recipes, 1):
            output += f"### {idx}. {recipe['title']}\n\n"
            
            # Create a table for ingredients
            output += "**Ingredients:**\n"
            output += "| Ingredient |\n"
            output += "|------------|\n"
            for ingredient in recipe['ingredients']:
                output += f"| {ingredient} |\n"
            output += "\n"
            
            # Display instructions and calorie estimate
            output += f"**Instructions:**\n{recipe['instructions']}\n\n"
            output += f"**Calorie Estimate:** {recipe['calorie_estimate']} kcal\n\n"
            
            # Show health score and prep time if available
            if recipe.get('health_score'):
                output += f"**Health Score:** {recipe['health_score']}/10\n\n"
            if recipe.get('prep_time'):
                output += f"**Prep Time:** {recipe['prep_time']}\n\n"
            
            output += "---\n\n"
    else:
        output += "No recipes could be generated.\n\n"

    # Quick Meal Section
    quick_meal = final_output.get("quick_meal")
    if quick_meal:
        output += "## ⚡ Quick & Healthy Meal (Under 15 min)\n\n"
        output += f"### {quick_meal.get('title', 'Quick Meal')}\n\n"
        
        output += "**Ingredients:**\n"
        output += "| Ingredient |\n"
        output += "|------------|\n"
        for ingredient in quick_meal.get('ingredients', []):
            output += f"| {ingredient} |\n"
        output += "\n"
        
        output += f"**Instructions:**\n{quick_meal.get('instructions', '')}\n\n"
        output += f"**Prep Time:** {quick_meal.get('prep_time', 'N/A')}\n\n"
        output += f"**Calorie Estimate:** {quick_meal.get('calorie_estimate', 'N/A')} kcal\n\n"
        output += f"**Health Score:** {quick_meal.get('health_score', 'N/A')}/10\n\n"
        output += f"**Why it's healthy:** {quick_meal.get('why_healthy', '')}\n\n"
        output += "---\n\n"

    # Health News Tips Section
    health_tips = final_output.get("health_news_tips")
    if health_tips:
        output += "## 📰 Health News Tips Applied\n\n"
        output += "| Headline | Tip | Source |\n"
        output += "|----------|-----|--------|\n"
        for tip in health_tips:
            headline = tip.get('headline', 'N/A')
            health_tip = tip.get('health_tip', tip.get('summary', 'N/A'))
            source = tip.get('source', 'N/A')
            output += f"| {headline} | {health_tip} | {source} |\n"
        output += "\n"

    return output

def format_analysis_output(final_output):
    """
    Formats nutritional analysis output into a table-based Markdown format,
    including health evaluation at the end.
    
    :param final_output: The JSON output from the NourishBotAnalysis workflow.
    :return: Formatted output as a Markdown string.
    """
    output = "## 🥗 Nutritional Analysis\n\n"
    
    # Basic dish information
    if dish := final_output.get('dish'):
        output += f"**Dish:** {dish}\n\n"
    if portion := final_output.get('portion_size'):
        output += f"**Portion Size:** {portion}\n\n"
    if est_cal := final_output.get('estimated_calories'):
        output += f"**Estimated Calories:** {est_cal} calories\n\n"
    if total_cal := final_output.get('total_calories'):
        output += f"**Total Calories:** {total_cal} calories\n\n"

    # Nutrient breakdown table
    output += "**Nutrient Breakdown:**\n\n"
    output += "| **Nutrient**       | **Amount** |\n"
    output += "|--------------------|------------|\n"
    
    nutrients = final_output.get('nutrients', {})
    # Display macronutrients
    for macro in ['protein', 'carbohydrates', 'fats']:
        if value := nutrients.get(macro):
            output += f"| **{macro.capitalize()}** | {value} |\n"
    
    # Display vitamins table if available
    vitamins = nutrients.get('vitamins', [])
    if vitamins:
        output += "\n**Vitamins:**\n\n"
        output += "| **Vitamin** | **%DV** |\n"
        output += "|-------------|--------|\n"
        for v in vitamins:
            name = v.get('name', 'N/A')
            dv = v.get('percentage_dv', 'N/A')
            output += f"| {name} | {dv} |\n"
    
    # Display minerals table if available
    minerals = nutrients.get('minerals', [])
    if minerals:
        output += "\n**Minerals:**\n\n"
        output += "| **Mineral** | **Amount** |\n"
        output += "|-------------|-----------|\n"
        for m in minerals:
            name = m.get('name', 'N/A')
            amount = m.get('amount', 'N/A')
            output += f"| {name} | {amount} |\n"
    
    # Append health evaluation at the end
    if health_eval := final_output.get('health_evaluation'):
        output += "\n**Health Evaluation:**\n\n"
        output += health_eval + "\n"
    
    return output


def analyze_food(image, dietary_restrictions, workflow_type, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper function for the Gradio interface.
    
    :param image: Uploaded image (PIL format)
    :param dietary_restrictions: Dietary restriction as a string (e.g., "vegan")
    :param workflow_type: Workflow type ("recipe" or "analysis")
    :return: Result from the NourishBot workflow.
    """
    
    image_path = os.path.abspath("uploaded_image.jpg")
    image.save(image_path)  # Save the uploaded image to an absolute path

    inputs = {
        'uploaded_image': image_path,
        'dietary_restrictions': dietary_restrictions,
        'workflow_type': workflow_type
    }
    
    # Initialize the appropriate crew instance based on workflow type
    if workflow_type == "recipe":
        crew_instance = NourishBotRecipeCrew(
            image_data=image_path,
            dietary_restrictions=dietary_restrictions
        )
    elif workflow_type == "analysis":
        crew_instance = NourishBotAnalysisCrew(
            image_data=image_path
        )
    else:
        return "Invalid workflow type. Choose 'recipe' or 'analysis'."

    # Run the crew workflow and get the result
    crew_obj = crew_instance.crew()
    crew_result = crew_obj.kickoff(inputs=inputs)

    if workflow_type == "recipe":
        # Extract structured JSON from output_json — prefer json_dict, fall back to pydantic
        recipe_data = {}
        if hasattr(crew_result, 'json_dict') and crew_result.json_dict:
            recipe_data = crew_result.json_dict
        elif hasattr(crew_result, 'pydantic') and crew_result.pydantic:
            recipe_data = crew_result.pydantic.model_dump()
        return format_recipe_output(recipe_data)
    elif workflow_type == "analysis":
        analysis_data = {}
        if hasattr(crew_result, 'json_dict') and crew_result.json_dict:
            analysis_data = crew_result.json_dict
        elif hasattr(crew_result, 'pydantic') and crew_result.pydantic:
            analysis_data = crew_result.pydantic.model_dump()
        return format_analysis_output(analysis_data)


def submit_feedback(recipe_name, rating, comment, dietary_restrictions):
    """Store user feedback about a recipe into the vector database."""
    if not recipe_name or not recipe_name.strip():
        return "⚠️ Please enter a recipe name."
    store_feedback(
        recipe_title=recipe_name.strip(),
        rating=int(rating),
        comment=comment.strip() if comment else "",
        dietary_restrictions=dietary_restrictions.strip() if dietary_restrictions else "",
    )
    return f"✅ Thank you! Feedback for **{recipe_name}** (Rating: {int(rating)}/5) has been saved. Future suggestions will learn from this!"


def refresh_health_news():
    """Fetch latest health news from trusted sources and store in vector DB."""
    articles = fetch_health_news(max_per_feed=3)
    if not articles:
        return "⚠️ Could not fetch news at this time. Please try again later."
    stored = process_and_store_news(articles)
    
    output = "## 📰 Latest Health & Nutrition News\n\n"
    output += "| # | Headline | Source |\n"
    output += "|---|----------|--------|\n"
    for i, article in enumerate(stored, 1):
        title = article.get("title", "N/A")
        source = article.get("source", "N/A")
        output += f"| {i} | {title} | {source} |\n"
    output += f"\n*{len(stored)} articles fetched and stored in memory. These will influence future recipe suggestions.*\n"
    return output


def get_recipe_choices():
    """Return current recipe names for feedback dropdown."""
    global _last_recipes
    return _last_recipes if _last_recipes else ["No recipes generated yet"]
    
# Define custom CSS for styling
css = """
.title {
    font-size: 1.5em !important; 
    text-align: center !important;
    color: #FFD700; 
}

.text {
    text-align: center;
}

.feedback-section {
    border: 2px solid #4CAF50;
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
}

.news-section {
    border: 2px solid #2196F3;
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
}
"""

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.color = '#eba93f';

    var text = 'Welcome to your AI NourishBot!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.1s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '0.9';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""
# Use a theme and custom CSS with Blocks
with gr.Blocks(theme=gr.themes.Citrus(), css=css, js=js) as demo:
    gr.Markdown("# How it works", elem_classes="title")
    gr.Markdown("Upload an image of your fridge content, enter your dietary restriction (if you have any!) and select a workflow type 'recipe' then click 'Analyze' to get recipe ideas.", elem_classes="text")
    gr.Markdown("Upload an image of a complete dish, leave dietary restriction blank and select a workflow type 'analysis' then click 'Analyze' to get nutritional insights.", elem_classes="text")
    gr.Markdown("NourishBot **learns from your feedback** and incorporates the **latest health news** into suggestions!", elem_classes="text")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Inputs", elem_classes="title")
            image_input = gr.Image(type="pil", label="Upload Image")
            dietary_input = gr.Textbox(label="Dietary Restrictions (optional)", placeholder="e.g., vegan")
            workflow_radio = gr.Radio(["recipe", "analysis"], label="Workflow Type")
            submit_btn = gr.Button("Analyze")
        
        with gr.Column(scale=2, min_width=600):
            gr.Examples(
                examples=[
                    ["examples/food-1.jpg", "vegan", "recipe"],
                    ["examples/food-2.jpg", "", "analysis"],
                    ["examples/food-3.jpg", "keto", "recipe"],
                    ["examples/food-4.jpg", "", "analysis"],
                ],
                inputs=[image_input, dietary_input, workflow_radio],
                label="Try an Example: Select one of the examples below to autofill the input section then click Analyze"
            )
            gr.Markdown("## Results will appear here...", elem_classes="title")
            result_display = gr.Markdown(
                "<div style='border: 1px solid #ccc; "
                "padding: 1rem; text-align: center; "
                "color: #666;'>No results yet</div>",
                height=500
            )

    # ---- Feedback Section ----
    gr.Markdown("---")
    gr.Markdown("## 💬 Recipe Feedback (Learning Agent)", elem_classes="title")
    gr.Markdown("Rate the recipes you received. NourishBot will learn from your feedback and personalize future suggestions!", elem_classes="text")

    with gr.Row(elem_classes="feedback-section"):
        with gr.Column(scale=1):
            fb_recipe_name = gr.Textbox(label="Recipe Name", placeholder="Enter the recipe name you want to rate")
            fb_rating = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Rating (1-5 stars)")
            fb_comment = gr.Textbox(label="Comment (optional)", placeholder="e.g., 'Too spicy' or 'Loved the flavors!'", lines=2)
            fb_dietary = gr.Textbox(label="Your Dietary Restrictions (optional)", placeholder="e.g., vegan, keto")
            fb_submit_btn = gr.Button("Submit Feedback", variant="primary")
        with gr.Column(scale=1):
            fb_result = gr.Markdown(
                "<div style='border: 1px solid #4CAF50; padding: 1rem; text-align: center; "
                "color: #666; border-radius: 8px;'>Submit feedback to help NourishBot learn your preferences!</div>"
            )

    fb_submit_btn.click(
        fn=submit_feedback,
        inputs=[fb_recipe_name, fb_rating, fb_comment, fb_dietary],
        outputs=fb_result
    )

    # ---- Health News Section ----
    gr.Markdown("---")
    gr.Markdown("## 📰 Health News Feed", elem_classes="title")
    gr.Markdown("Fetch the latest health & nutrition news from trusted sources (WHO, Harvard Health, Medical News Today, NHS). News is stored in memory and influences future recipe suggestions!", elem_classes="text")

    with gr.Row(elem_classes="news-section"):
        with gr.Column():
            news_refresh_btn = gr.Button("🔄 Refresh Health News", variant="secondary")
            news_display = gr.Markdown(
                "<div style='border: 1px solid #2196F3; padding: 1rem; text-align: center; "
                "color: #666; border-radius: 8px;'>Click 'Refresh Health News' to fetch the latest articles.</div>",
                height=400
            )

    news_refresh_btn.click(
        fn=refresh_health_news,
        inputs=[],
        outputs=news_display
    )

    submit_btn.click(
        fn=analyze_food,
        inputs=[image_input, dietary_input, workflow_radio],
        outputs=result_display
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5000)
