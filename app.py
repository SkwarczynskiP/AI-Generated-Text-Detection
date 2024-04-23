import gradio as gr
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Models included within the interface
models = ["bert-base-uncased", "roberta-base"]

# Datasets included within the interface
datasets = ["No Dataset Finetuning",
            "vedantgaur/GPTOutputs-MWP - AI Data Only",
            "vedantgaur/GPTOutputs-MWP - Human Data Only",
            "vedantgaur/GPTOutputs-MWP - Both AI and Human Data",
            "dmitva/human_ai_generated_text - Both AI and Human Data"]

# Mapping of user-selected model and dataset to actual model name on Hugging Face
model_mapping = {
    ("bert-base-uncased", "No Dataset Finetuning"): "bert-base-uncased",
    ("bert-base-uncased", "vedantgaur/GPTOutputs-MWP - AI Data Only"): "SkwarczynskiP/bert-base-uncased-finetuned-vedantgaur-AI-generated",
    ("bert-base-uncased", "vedantgaur/GPTOutputs-MWP - Human Data Only"): "SkwarczynskiP/bert-base-uncased-finetuned-vedantgaur-human-generated",
    ("bert-base-uncased", "vedantgaur/GPTOutputs-MWP - Both AI and Human Data"): "SkwarczynskiP/bert-base-uncased-finetuned-vedantgaur-AI-and-human-generated",
    ("bert-base-uncased", "dmitva/human_ai_generated_text - Both AI and Human Data"): "SkwarczynskiP/bert-base-uncased-finetuned-dmitva-AI-and-human-generated",
    ("roberta-base", "No Dataset Finetuning"): "roberta-base",
    ("roberta-base", "vedantgaur/GPTOutputs-MWP - AI Data Only"): "SkwarczynskiP/roberta-base-finetuned-vedantgaur-AI-generated",
    ("roberta-base", "vedantgaur/GPTOutputs-MWP - Human Data Only"): "SkwarczynskiP/roberta-base-finetuned-vedantgaur-human-generated",
    ("roberta-base", "vedantgaur/GPTOutputs-MWP - Both AI and Human Data"): "SkwarczynskiP/roberta-base-finetuned-vedantgaur-AI-and-human-generated",
    ("roberta-base", "dmitva/human_ai_generated_text - Both AI and Human Data"): "SkwarczynskiP/roberta-base-finetuned-dmitva-AI-and-human-generated"
}

# Example text included within the interface
exampleText = [
    "Certainly! New York City is a vibrant and dynamic place with an abundance of cool and unusual activities. Here are some recommendations to make your NYC experience memorable: 1. Visit the High Line, a unique elevated park built on a historic freight rail line. 2. Explore the quirky shops and restaurants in the East Village. 3. Take a ferry to Governors Island for stunning views of the city skyline. 4. Attend a live performance at the Upright Citizens Brigade Theatre for some laughs. 5. Check out the street art in Bushwick, Brooklyn. Have a great time in NYC!",
    "Throughout history, numerous remarkable women have left an indelible mark on society. Here are some influential women and their notable achievements: 1. Marie Curie - Nobel Prize-winning physicist and chemist who discovered radium and polonium. 2. Rosa Parks - Civil rights activist known for her pivotal role in the Montgomery bus boycott. 3. Malala Yousafzai - Youngest Nobel Prize laureate for her advocacy",
    "I sometimes get frustrated when customers know it’s cold outside but don’t have their mobile app ready to be scanned or their card out of their very deep purse. They have to deep dive to get it, and I end up standing there, wind gushing in my face, unable to breathe. Hence why I love the people who have everything prepared by the time they get to the window – truly the greatest humans on earth. Like, let’s just be prepared by the time your car is at the window: debit/credit card out, money ready, mobile app open to scan. It makes it easier and faster for both of us.",
    "There needs to be a distinction between performance improvement vs. role improvement. These two can go hand in hand, however they are separate. Tyrese Maxey is a perfect case of these being separate. In the 2022-23 season, Maxey averaged 28.9 PPG in his 10 games without Embid. This season that number was 26.3 PPG in 34 games without Embid. Both figures, are amazing and it’s clear Maxey is a top tier scoring option. But did his performance truly improve? When you compare season to season, his PPG increased from 20.3 to 25.9.",
]

# Example models and datasets included within the interface
exampleModels = ["bert-base-uncased", "roberta-base"]

# Example datasets included within the interface
exampleDatasets = ["No Dataset Finetuning",
            "vedantgaur/GPTOutputs-MWP - AI Data Only",
            "vedantgaur/GPTOutputs-MWP - Human Data Only",
            "vedantgaur/GPTOutputs-MWP - Both AI and Human Data",
            "dmitva/human_ai_generated_text - Both AI and Human Data"]

examples = [[random.choice(exampleModels), random.choice(exampleDatasets), example] for example in exampleText]


def detect_ai_generated_text(model: str, dataset: str, text: str) -> list:
    # Get the fine-tuned model using mapping
    finetuned_model = model_mapping.get((model, dataset))

    # Load the specific fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_model)

    # Classify the input based on the fine-tuned model
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    result = classifier(text)

    # Get the label and score
    label = "AI-generated" if result[0]['label'] == 'LABEL_1' else "Human-written"
    score = result[0]['score']

    # Create HTML for the colored bars
    ai_score = score if label == "AI-generated" else 1 - score
    human_score = 1 - ai_score
    ai_bar = f'<div style="margin-bottom: 1em;"><div style="background-color: #ff7f7f; width: {ai_score * 100}%; height: 20px;"></div><div>AI-generated</div></div>'
    human_bar = f'<div style="margin-bottom: 1em;"><div style="background-color: #7f7fff; width: {human_score * 100}%; height: 20px;"></div><div>Human-written</div></div>'
    
    # Scale down the size of the outputted text
    scaled_label = f'{label} - Confidence Level: {score * 100:.2f}%'

    return [scaled_label, ai_bar, human_bar]


interface = gr.Interface(
    fn=detect_ai_generated_text,
    inputs=[
        gr.Dropdown(choices=models, label="Model"),
        gr.Dropdown(choices=datasets, label="Dataset"),
        gr.Textbox(lines=5, label="Input Text")
    ],
    outputs=[
        gr.HTML(label="Output"),
        gr.HTML(label="AI-generated"),
        gr.HTML(label="Human-written")
    ],
    examples=examples,
    title="AI Generated Text Detection"
)

if __name__ == "__main__":
    interface.launch()