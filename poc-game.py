from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate

"""Proof of concept for detective game, player is supposed to get confession
from LLM playing a convict

"""


class CharacterCard:
    """
    Represents a character card with information about a fictional character.
    """
    def __init__(self, name, surname, age, backstory, crime,
                 motive, alibi, inconsistencies, unique_traits):
        self.name = name
        self.surname = surname
        self.age = age
        self.backstory = backstory
        self.crime = crime
        self.motive = motive
        self.alibi = alibi
        self.inconsistencies = inconsistencies
        self.unique_traits = unique_traits

    def get_info(self):
        """
        Returns a concatenated string with the information
        of the character card.
        Each attribute is on a separate line.

        Returns:
            str: Concatenated string with each attribute on a separate line.
        """
        info = "Name: {}\n".format(self.name)
        info += "Surname: {}\n".format(self.surname)
        info += "Age: {}\n".format(self.age)
        info += "Backstory: {}\n".format(self.backstory)
        info += "Crime: {}\n".format(self.crime)
        info += "Motive: {}\n".format(self.motive)
        info += "Alibi: {}\n".format(self.alibi)
        info += "Inconsistencies: {}\n".format(self.inconsistencies)
        info += "Unique Traits: {}\n".format(self.unique_traits)
        return info

    def update_backstory(self, new_backstory):
        """
        Updates the character's backstory with the provided information.

        Args:
            new_backstory (str): The new backstory to be set for the character.
        """
        self.backstory = new_backstory

    def update_crime(self, new_crime):
        """
        Updates the character's crime with the provided information.

        Args:
            new_crime (str): The new crime to be set for the character.
        """
        self.crime = new_crime
      
    def update_motive(self, new_motive):
        """
        Updates the character's motive with the provided information.

        Args:
            new_motive (str): The new motive to be set for the character.
        """
        self.motive = new_motive


template = """You are a convict, interrogated by the police.
Below there are details about yourself and conversation history.
You're supposed to deny all allegations unless
interrogator presents sufficient proof,
you can modify alibi in order to maintain innocence.
You're supposed to confess guilt only after your initial alibi has been
proven completely false or if Interrogator
mentioned at least two inconsistencies.
Remember to include unique traits in your writing style.

{character_card}

{chat_history}
"""

prompt = PromptTemplate(
    input_variables=["character_card", "chat_history"],
    template=template
)

llm_chain = LLMChain(
    llm=OpenAI(openai_api_key="", temperature=0.366),
    prompt=prompt,
    verbose=True
)

chat_history = ""
human_input = ""
test_card = CharacterCard(
    name="Jack",
    surname="Sparrow",
    age=44,
    backstory="You're a pirate captain of ship called Black Pearl, you travel the Carribean",
    crime="You assaulted smith's apprentice with a sword, you beheaded him leaving a bloody sword",
    motive="Smith's apprentice threatened to alert authorities",
    alibi="You were passed out in local tavern, numerous friends can vouch for you",
    inconsistencies="Nobody would vouch for you, your sword was left at the scene of crime and there's blood all over you",
    unique_traits="You speak like a pirate"
)

# print suspect card
while human_input != 'end':
    human_input = input("line: ")
    chat_history += ("\n Interrogator: {} \n Chatbot:".format(human_input))
    response = llm_chain.predict(
        character_card=test_card.get_info(),
        chat_history=chat_history)
    print(response)
    chat_history += (response)
