import streamlit as st
import wandb
import weave
import asyncio
from weave import Model, Evaluation, Dataset
import json
from openai import OpenAI

# We call init to begin capturing data in the project, intro-example.
PROJECT = "class_finserv_classify"
weave.init(PROJECT)


# We create a model class with one predict function.
# All inputs, predictions and parameters are automatically captured for easy inspection.
class ExtractFinancialDetailsModel(Model):
    system_message: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7

    @weave.op()
    def predict(self, article: str) -> dict:
        from openai import OpenAI

        client = OpenAI()
        self.system_message = sys_prompt
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": article},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        model_output = response.choices[0].message.content
        return json.loads(model_output)


# CNBC News Articles
articles = [
    "Novo Nordisk and Eli Lilly rival soars 32 percent after promising weight loss drug results Shares of Denmarks Zealand Pharma shot 32 percent higher in morning trade, after results showed success in its liver disease treatment survodutide, which is also on trial as a drug to treat obesity. The trial “tells us that the 6mg dose is safe, which is the top dose used in the ongoing [Phase 3] obesity trial too,” one analyst said in a note. The results come amid feverish investor interest in drugs that can be used for weight loss.",
    "Berkshire shares jump after big profit gain as Buffetts conglomerate nears $1 trillion valuation Berkshire Hathaway shares rose on Monday after Warren Buffetts conglomerate posted strong earnings for the fourth quarter over the weekend. Berkshires Class A and B shares jumped more than 1.5%, each. Class A shares are higher by more than 17% this year, while Class B have gained more than 18%. Berkshire was last valued at $930.1 billion, up from $905.5 billion where it closed on Friday, according to FactSet. Berkshire on Saturday posted fourth-quarter operating earnings of $8.481 billion, about 28 percent higher than the $6.625 billion from the year-ago period, driven by big gains in its insurance business. Operating earnings refers to profits from businesses across insurance, railroads and utilities. Meanwhile, Berkshires cash levels also swelled to record levels. The conglomerate held $167.6 billion in cash in the fourth quarter, surpassing the $157.2 billion record the conglomerate held in the prior quarter.",
    "Highmark Health says its combining tech from Google and Epic to give doctors easier access to information Highmark Health announced it is integrating technology from Google Cloud and the health-care software company Epic Systems. The integration aims to make it easier for both payers and providers to access key information they need, even if its stored across multiple points and formats, the company said. Highmark is the parent company of a health plan with 7 million members, a provider network of 14 hospitals and other entities",
    "Rivian and Lucid shares plunge after weak EV earnings reports Shares of electric vehicle makers Rivian and Lucid fell Thursday after the companies reported stagnant production in their fourth-quarter earnings after the bell Wednesday. Rivian shares sank about 25 percent, and Lucids stock dropped around 17 percent. Rivian forecast it will make 57,000 vehicles in 2024, slightly less than the 57,232 vehicles it produced in 2023. Lucid said it expects to make 9,000 vehicles in 2024, more than the 8,428 vehicles it made in 2023.",
    "Mauritius blocks Norwegian cruise ship over fears of a potential cholera outbreak Local authorities on Sunday denied permission for the Norwegian Dawn ship, which has 2,184 passengers and 1,026 crew on board, to access the Mauritius capital of Port Louis, citing “potential health risks.” The Mauritius Ports Authority said Sunday that samples were taken from at least 15 passengers on board the cruise ship. A spokesperson for the U.S.-headquartered Norwegian Cruise Line Holdings said Sunday that 'a small number of guests experienced mild symptoms of a stomach-related illness' during Norwegian Dawns South Africa voyage.",
    "Intuitive Machines lands on the moon in historic first for a U.S. company Intuitive Machines Nova-C cargo lander, named Odysseus after the mythological Greek hero, is the first U.S. spacecraft to soft land on the lunar surface since 1972. Intuitive Machines is the first company to pull off a moon landing — government agencies have carried out all previously successful missions. The companys stock surged in extended trading Thursday, after falling 11 percent in regular trading.",
    "Lunar landing photos: Intuitive Machines Odysseus sends back first images from the moon Intuitive Machines cargo moon lander Odysseus returned its first images from the surface. Company executives believe the lander caught its landing gear sideways in the moons surface while touching down and tipped over. Despite resting on its side, the companys historic IM-1 mission is still operating on the moon.",
]

labels = [
    {
        "company_name": "Zealand Pharma",
        "company_ticker": "0NZU-GB",
        "document_sentiment": "Positive",
    },
    {
        "company_name": "Berkshire Hathaway",
        "company_ticker": ["BRK.A", "BRK.B"],
        "document_sentiment": "Positive",
    },
    {
        "company_name": "Highmark Health",
        "company_ticker": "N/A",
        "document_sentiment": "Positive",
    },
    {
        "company_name": ["Rivian", "Lucid"],
        "company_ticker": ["RIVN", "LCID"],
        "document_sentiment": "Negative",
    },
    {
        "company_name": "Norwegian Cruise Line Holdings",
        "company_ticker": "NCLH",
        "document_sentiment": "Negative",
    },
    {
        "company_name": "Intuitive Machines",
        "company_ticker": "LUNR",
        "document_sentiment": "Positive",
    },
    {
        "company_name": "Intuitive Machines",
        "company_ticker": "LUNR",
        "document_sentiment": "Positive",
    },
]

# Create the evaluation dataset
dataset = Dataset(
    name="Financial Services Evaluation Data",
    description="Financial services articles with actual labels to use for model evaluation.",
    rows=[
        {"id": str(i), "article": articles[i], "labeled_actuals": labels[i]}
        for i in range(len(articles))
    ],
)
dataset_ref = weave.publish(dataset, "finserv-eval-data")


# define the hallucination model
class DetermineHallucinationModel(Model):
    system_message: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7

    @weave.op()
    def predict(self, text_to_analyze: str) -> dict:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text_to_analyze},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        model_output = response.choices[0].message.content
        try:
            return json.loads(model_output)
        except Exception:
            return {}


# We define four scoring functions to compare our model predictions with a ground truth label.
@weave.op()
def name_score(model_output: dict, labeled_actuals: dict) -> dict:
    correct = True
    for label in labeled_actuals["company_name"]:
        if isinstance(model_output.get("company_name"), list):
            modelOutput = " ".join(str(item) for item in model_output["company_name"])
            correct = correct and label.lower() in modelOutput.lower()
        else:
            correct = correct and label.lower() in model_output.get("company_name", "").lower()
    return {"correct": correct}


@weave.op()
def ticker_score(model_output: dict, labeled_actuals: dict) -> dict:
    correct = True
    for label in labeled_actuals["company_ticker"]:
        if isinstance(model_output.get("company_ticker"), list):
            modelOutput = " ".join(str(item) for item in model_output["company_ticker"])
            correct = correct and label.lower() in modelOutput.lower()
        else:
            correct = (
                correct and label.lower() in model_output.get("company_ticker", "").lower()
            )
    return {"correct": correct}


@weave.op()
def sentiment_score(model_output: dict, labeled_actuals: dict) -> dict:
    return {
        "correct": model_output.get("document_sentiment", "").lower()
        == labeled_actuals["document_sentiment"].lower()
    }


@weave.op()
def hallucination_score(model_output: dict) -> dict:
    hallucination_calculation_model = DetermineHallucinationModel(
        system_message="You are in charge of determining if the text submitted is the result of LLM hallucination or not. Your task is to respond with a JSON dictionary including a single hallucanation score. The hallucination score should be a float from 0 to 100, where 100 is more likely and 0 is less likely that the text is hallucination or not."
    )
    return hallucination_calculation_model.predict(model_output.get("summary", ""))


@weave.op()
def example_to_model_input(example: dict) -> str:
    # example is a row from the Dataset, the output of this function should be the input to model.predict.
    return example["article"]


# Finally, we run an evaluation of this model.
# This will generate a prediction for each input example, and then score it with each scoring function.
evaluation = Evaluation(
    dataset=dataset,
    scorers=[name_score, ticker_score, sentiment_score, hallucination_score],
)

model_names = ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"]

system_messages = [
    "You will be provided with financial news and your task is to parse it one JSON dictionary with company_ticker, company_name, document_sentiment, and summary as keys. The summary should be a two sentance summary of the news and responses should all be strings.",
    "You will be provided with financial news and your task is to parse it one JSON dictionary with company_ticker, company_name, document_sentiment, and summary as keys. The summary should be a two sentance summary of the news, responses should all be strings, there could be multiple responses per article, and the company ticker should be the stock exchange ticker symbol.",
]

## Streamlit App
st.set_page_config(page_title="FinServ – Classify")
st.header("Tweak the prompt & temperature")
user_name = st.text_input(
    label="Your Name",
    value="",
    placeholder="CVP",
    disabled=False,
    label_visibility="visible",
)
sys_prompt = st.text_area(
    label="System Prompt",
    key="input",
    label_visibility="hidden",
    value="You will be provided with financial news and your task is to parse it one JSON dictionary with company_ticker, company_name, document_sentiment, and summary as keys. The summary should be a two sentence summary of the news, responses should all be strings, there could be multiple responses per article, and the company ticker should be the stock exchange ticker symbol. You always start and end the answer with a friendly pirate greeting.",
    height=300,
)
temperature = st.slider(
    label="Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
)
submit = st.button("Run")

# if submit is clicked
if submit:
    if not user_name:
        st.error('Your name is required to submit')
    # call the model
    api = wandb.Api()
    entity = api.default_entity

    # call the model
    for model_name in model_names:
        # We create our model with our system prompt.
        model = ExtractFinancialDetailsModel(
            model_name=model_name, system_message=sys_prompt, temperature=temperature
        )
        with weave.attributes({"user_name": user_name}):
            results = asyncio.run(evaluation.evaluate(model))
            print(results)

            st.header("Results")
            st.json(results)
            st.write(
                f"View results at: https://wandb.ai/{entity}/{PROJECT}/weave"
            )
