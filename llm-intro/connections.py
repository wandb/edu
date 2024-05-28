import streamlit as st
import weave
import asyncio
import wandb
from weave import Model, Evaluation, Dataset
import json
from openai import OpenAI

# We call init to begin capturing data in the project, intro-example.
PROJECT = "dev_class_play_connections"
weave.init(PROJECT)


# We create a model class with one predict function.
# All inputs, predictions and parameters are automatically captured for easy inspection.
class ConnectionsModel(Model):
    system_message: str
    model_name: str = "gpt-3.5-turbo-1106"

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
            temperature=0.7,
        )
        model_output = response.choices[0].message.content
        return model_output


# True Connections
connections = [
    "TIME, LOAFER, QUEUE, ESSENCE, MILE, PUMP, FOOT, YARD, LEAGUE, ARE, WHY, SEA, US, PEOPLE, SNEAKER, BOOT",
    "POWDER, SIN, WONDER, FUTURE, BABY, CHANCE, CUBE, SEA, BOARDWALK, SISTER, GO, ROYAL, ICE JAIL, COMMON, MIDNIGHT, Q-TIP",
    "THRONE, THIGH, CAN, CALF, JOEY, KNEE, HEAD, STAR, SILVER, JELLY, CUB, KID, CRAY, SHIN, JOHN, ANKLE",
    "CHERRY, KING, CHESS, LOCKSMITH, TRUCK, CRYPTOGRAPHY, PIANO, QUEEN, SIGN, BACKGAMMON, FLORIDA, STOP GO, FIRE FULL, RUBY, CHECKERS, TWIN",
    "TANK, CRICKET, MANTIS, FENCING, ANT, HALTER, SQUASH, BEETLE, CAMI, CARROT, TERMITE, BEET, CORN, ONION, POLO, TEE",
    "GOAT, TWINS, GEORGIA, SWALLOW, LEMON, JAY, KIWI, SCALES, DATE, CRANE, TURKEY, ORANGE, CHAD, FISH, JORDAN, TOGO",
    "GINGER, SUGAR, ANGEL, CUMIN, CARDAMOM, BABY, BOO, CORIANDER, SCARY, CLOVE, POSH, SWEETIE, HONEY, PEGASUS, AIRPLANE, BIRD",
    "POD, COT, PACK, LUST, SCHOOL, SNAIL, GREED, SIN, TORTOISE, SLOTH, SEC, LORIS, FLOCK, PRIDE, TAN, ENVY",
    "CHI, VIRGIN, GNOME, GNAT, NU, SPIRIT, GNU, FRONTIER, KNEW, UNITED, BETA, GNOCCHI, NEW, IOTA, GNAW, DELTA",
    "ACCORD, MURPHY, TRUNDLE, PILOT, SONIC, CIVIC, CRASH, WRIGHT, MARX, PASSPORT, BUNK, MARIO, LINK, WARNER, CANOPY, JONAS",
    "JONES, BROWN, ROD, CAW, SPARROW, TACKLE, CHIRP, HOOK, LURE, TWEET, VIOLET, CLUCK, REEL, PINK, SILVER, TURQUOISE",
]

labels = [
    """
    BOOT, LOAFER, PUMP, SNEAKER,
    FOOT, LEAGUE, MILE, YARD,
    ESSENCE, PEOPLE, TIME, US,
    ARE, QUEUE, SEA, WHY
    """,
    """
    BOARDWALK, CHANCE, GO, JAIL,
    BABY, MIDNIGHT, POWDER, ROYAL,
    COMMON, FUTURE, ICE CUBE, Q-TIP,
    SEA, SIN, SISTER, WONDER
    """,
    """
    ANKLE, KNEE, SHIN, THIGH,
    CALF, CUB, JOEY, KID,
    CAN, HEAD, JOHN, THRONE,
    CRAY, JELLY, SILVER, STAR
    """,
    """
    BACKGAMMON, CHECKERS, CHESS, GO,
    FULL, KING, QUEEN, TWIN,
    CHERRY, FIRE TRUCK, RUBY, STOP SIGN,
    CRYPTOGRAPHY, FLORIDA, LOCKSMITH, PIANO
    """,
    """
    CRICKET, FENCING, POLO, SQUASH,
    CAMI, HALTER, TANK, TEE,
    BEET, CARROT, CORN, ONION,
    ANT, BEETLE, MANTIS, TERMITE
    """,
    """
    DATE, KIWI, LEMON, ORANGE,
    CHAD, GEORGIA, JORDAN, TOGO,
    CRANE, JAY, SWALLOW, TURKEY,
    FISH, GOAT, SCALES, TWINS
    """,
    """
    CARDAMOM, CLOVE, CORIANDER, CUMIN,
    BOO, HONEY, SUGAR, SWEETIE,
    AIRPLANE, ANGEL, BIRD, PEGASUS,
    BABY, GINGER, POSH, SCARY
    """,
    """
    FLOCK, PACK, POD, SCHOOL,
    ENVY, GREED, LUST, PRIDE,
    LORIS, SLOTH, SNAIL, TORTOISE,
    COT, SEC, SIN, TAN
    """,
    """
    FRONTIER, SPIRIT, UNITED, VIRGIN,
    BETA, CHI, DELTA, IOTA,
    GNAT, GNAW, GNOCCHI, GNOME,
    GNU, KNEW, NEW, NU
    """,
    """
    BUNK, CANOPY, MURPHY, TRUNDLE,
    JONAS, MARX, WARNER, WRIGHT,
    ACCORD, CIVIC, PASSPORT, PILOT,
    CRASH, LINK, MARIO, SONIC
    """,
    """
    CAW, CHIRP, CLUCK, TWEET,
    BROWN, PINK, TURQUOISE, VIOLET,
    LURE, REEL, ROD, TACKLE,
    HOOK, JONES, SILVER, SPARROW
    """,
]

# Create the evaluation dataset
dataset = Dataset(
    name="Connections Data",
    description="NYT Connections game data",
    rows=[
        {"id": str(i), "article": connections[i], "labeled_actuals": labels[i]}
        for i in range(len(connections))
    ],
)
dataset_ref = weave.publish(dataset, "connections-eval-data")


# We define four scoring functions to compare our model predictions with a ground truth label.
@weave.op()
def correctness_score(model_output: str, labeled_actuals: str) -> dict:
    print("labels")
    print("____________________")
    print(labeled_actuals)
    print("____________________")
    print("model_output")
    print("____________________")
    print(model_output)
    print("____________________")

    def get_group_set(string):
        group_set = []
        groups = string.strip().split("\n")
        for group in groups:
            words = sorted([word.strip() for word in group.split(",") if word.strip()])
            group_set.append(words)
        return group_set

    label_groups = get_group_set(labeled_actuals)
    test_groups = get_group_set(model_output)

    score = 0
    for group in label_groups:
        if group in test_groups:
            score += 1

    print("label_groups")
    print(label_groups)
    print("____________________")
    print("model_output_groups")
    print(test_groups)
    print("____________________")
    return {
        "model_output": model_output,
        "labeled_actuals": labeled_actuals,
        "score": score,
    }


@weave.op()
def format_score(model_output: str) -> bool:
    groups = model_output.strip().split("\n")
    if len(groups) != 4:
        return False
    for group in groups:
        words = group.split(",")
        if len(words) != 4:
            return False
    return True


@weave.op()
def example_to_model_input(example: dict) -> str:
    # example is a row from the Dataset, the output of this function should be the input to model.predict.
    return example["article"]


# Finally, we run an evaluation of this model.
# This will generate a prediction for each input example, and then score it with each scoring function.
evaluation = Evaluation(dataset=dataset, scorers=[correctness_score, format_score])

model_names = ["gpt-3.5-turbo"]

## Streamlit App
st.set_page_config(page_title="FinServ – Classify")
st.header("Tweak the prompt")
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
    value="""You are an expert Connections player.
The aim of the game is to find groups of four items that share something in common.
The categories given are designed to be more distinct in nature than just “names”, “verbs” or “5-letter-words“. Each puzzle has exactly one solution. Watch out for words that seem to belong to multiple categories!

HOW TO PLAY
Given 16 words: SLEET, RAIN, NETS, RACECAR, SNOW, MOM, BUCKS, TAB, SHIFT, LEVEL, OPTION, RETURN, HAIL, JAZZ, KAYAK, HEAT

Return the items organized in groups of four that belong in a category. The 4 categories are separated by a newline like so:
BUCKS, HEAT, JAZZ, NETS
HAIL, RAIN, SLEET, SNOW
OPTION, RETURN, SHIFT, TAB
KAYAK, LEVEL, MOM, RACECAR

CATEGORY EXAMPLES
FISH: Bass, Flounder, Salmon, Trout
FIRE ___: Ant, Drill, Island, Opal
Palindromes: Kayak, Level, Mom, Racecar
Letter Homophones: Are, Quque, Sea, Why

Let's go!
""",
    height=300,
)
submit = st.button("Run")

# if submit is clicked
if submit:
    if not user_name:
        st.error("Your name is required to submit")
    # call the model
    api = wandb.Api()
    entity = api.default_entity

    for model_name in model_names:
        # We create our model with our system prompt.
        model = ConnectionsModel(model_name=model_name, system_message=sys_prompt)
        with weave.attributes({"user_name": user_name}):
            response = asyncio.run(evaluation.evaluate(model))
            print("response_______________")
            print(response)
            print("_______________")

            st.header("Results")
            st.json(response)
            st.write(
                f"View detailed results at: https://wandb.ai/{entity}/{PROJECT}/weave"
            )
