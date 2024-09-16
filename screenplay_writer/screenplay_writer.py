import re, os
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
# from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI



load_dotenv()


defalut_llm = ChatOpenAI(openai_api_base=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
                        openai_api_key=os.environ.get("OPENAI_API_KEY"),
                        temperature=0.1,                        
                        model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                        top_p=0.3)


# Use Path for file locations
current_dir = Path.cwd()
agents_config_path = current_dir / "config" / "agents.yaml"
tasks_config_path = current_dir / "config" / "tasks.yaml"

# Load YAML configuration files
with open(agents_config_path, "r") as file:
    agents_config = yaml.safe_load(file)

with open(tasks_config_path, "r") as file:
    tasks_config = yaml.safe_load(file)

## Define Agents
spamfilter = Agent( 
    role=agents_config["spamfilter"]['role'], 
    goal=agents_config["spamfilter"]['goal'], 
    backstory=agents_config["spamfilter"]['backstory'], 
    allow_delegation=False, 
    verbose=True, 
    llm=defalut_llm
)

analyst = Agent( 
    role=agents_config["analyst"]['role'], 
    goal=agents_config["analyst"]['goal'], 
    backstory=agents_config["analyst"]['backstory'], 
    allow_delegation=False, 
    verbose=True, 
    llm=defalut_llm
)

scriptwriter = Agent( 
    role=agents_config["scriptwriter"]['role'], 
    goal=agents_config["scriptwriter"]['goal'], 
    backstory=agents_config["scriptwriter"]['backstory'], 
    allow_delegation=False, 
    verbose=True, 
    llm=defalut_llm
)

formatter = Agent( 
    role=agents_config["formatter"]['role'], 
    goal=agents_config["formatter"]['goal'], 
    backstory=agents_config["formatter"]['backstory'], 
    allow_delegation=False, 
    verbose=True, 
    llm=defalut_llm
)

scorer = Agent( 
    role=agents_config["scorer"]['role'], 
    goal=agents_config["scorer"]['goal'], 
    backstory=agents_config["scorer"]['backstory'], 
    allow_delegation=False, 
    verbose=True, 
    llm=defalut_llm
)



# this is one example of a public post in the newsgroup alt.atheism
# try it out yourself by replacing this with your own email thread or text or ...
discussion = """From: keith@cco.caltech.edu (Keith Allan Schneider)
Subject: Re: <Political Atheists?
Organization: California Institute of Technology, Pasadena
Lines: 50
NNTP-Posting-Host: punisher.caltech.edu

bobbe@vice.ICO.TEK.COM (Robert Beauchaine) writes:

>>I think that about 70% (or so) people approve of the
>>death penalty, even realizing all of its shortcomings.  Doesn't this make
>>it reasonable?  Or are *you* the sole judge of reasonability?
>Aside from revenge, what merits do you find in capital punishment?

Are we talking about me, or the majority of the people that support it?
Anyway, I think that "revenge" or "fairness" is why most people are in
favor of the punishment.  If a murderer is going to be punished, people
that think that he should "get what he deserves."  Most people wouldn't
think it would be fair for the murderer to live, while his victim died.

>Revenge?  Petty and pathetic.

Perhaps you think that it is petty and pathetic, but your views are in the
minority.


keith
"""

oo_discussion = """From: keith@cco.caltech.edu (Keith Allan Schneider)
Subject: Re: <Political Atheists?
Organization: California Institute of Technology, Pasadena
Lines: 50
NNTP-Posting-Host: punisher.caltech.edu

bobbe@vice.ICO.TEK.COM (Robert Beauchaine) writes:

>>I think that about 70% (or so) people approve of the
>>death penalty, even realizing all of its shortcomings.  Doesn't this make
>>it reasonable?  Or are *you* the sole judge of reasonability?
>Aside from revenge, what merits do you find in capital punishment?

Are we talking about me, or the majority of the people that support it?
Anyway, I think that "revenge" or "fairness" is why most people are in
favor of the punishment.  If a murderer is going to be punished, people
that think that he should "get what he deserves."  Most people wouldn't
think it would be fair for the murderer to live, while his victim died.

>Revenge?  Petty and pathetic.

Perhaps you think that it is petty and pathetic, but your views are in the
minority.

>We have a local televised hot topic talk show that very recently
>did a segment on capital punishment.  Each and every advocate of
>the use of this portion of our system of "jurisprudence" cited the
>main reason for supporting it:  "That bastard deserved it".  True
>human compassion, forgiveness, and sympathy.

Where are we required to have compassion, forgiveness, and sympathy?  If
someone wrongs me, I will take great lengths to make sure that his advantage
is removed, or a similar situation is forced upon him.  If someone kills
another, then we can apply the golden rule and kill this person in turn.
Is not our entire moral system based on such a concept?

Or, are you stating that human life is sacred, somehow, and that it should
never be violated?  This would sound like some sort of religious view.
 
>>I mean, how reasonable is imprisonment, really, when you think about it?
>>Sure, the person could be released if found innocent, but you still
>>can't undo the imiprisonment that was served.  Perhaps we shouldn't
>>imprision people if we could watch them closely instead.  The cost would
>>probably be similar, especially if we just implanted some sort of
>>electronic device.
>Would you rather be alive in prison or dead in the chair?  

Once a criminal has committed a murder, his desires are irrelevant.

And, you still have not answered my question.  If you are concerned about
the death penalty due to the possibility of the execution of an innocent,
then why isn't this same concern shared with imprisonment.  Shouldn't we,
by your logic, administer as minimum as punishment as possible, to avoid
violating the liberty or happiness of an innocent person?

keith
"""


# Filter out spam and vulgar posts
task0 = Task(
    description=tasks_config["task0"]["description"].format(discussion=discussion),
    expected_output=tasks_config["task0"]["expected_output"],
    agent=spamfilter,
)

crew = Crew(
    agents=[spamfilter],
    tasks=[task0],
    verbose=True,  # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

inputs = {'discussion': discussion}
result = crew.kickoff(inputs)
# result = crew.kickoff()

print("===================== end result from crew ===================================")
print(result)

# Accessing the task output
task_output = task0.output

print(f"Task Description: {task_output.description}")
print(f"Task Summary: {task_output.summary}")
print(f"Raw Output: {task_output.raw}")
if task_output.json_dict:
    print(f"JSON Output: {json.dumps(task_output.json_dict, indent=2)}")
if task_output.pydantic:
    print(f"Pydantic Output: {task_output.pydantic}")


# process post with a crew of agents, ultimately delivering a well formatted dialogue
task1 = Task(
    description=tasks_config["task1"]["description"].format(discussion=discussion),
    expected_output=tasks_config["task1"]["expected_output"],
    agent=analyst,
)

task2 = Task(
    description=tasks_config["task2"]["description"],
    expected_output=tasks_config["task2"]["expected_output"],
    agent=scriptwriter,
)

task3 = Task(
    description=tasks_config["task3"]["description"],
    expected_output=tasks_config["task3"]["expected_output"],
    agent=formatter,
)
crew = Crew(
    agents=[analyst, scriptwriter, formatter],
    tasks=[task1, task2, task3],
    verbose=True,  # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

inputs = {'discussion': discussion}
result = crew.kickoff(inputs)

print("===================== end result from crew ===================================")
print(result)



# print("===================== score ==================================================")
# task4 = Task(
#     description=tasks_config["task4"]["description"],
#     expected_output=tasks_config["task4"]["expected_output"],
#     agent=scorer,
# )

# score = task4.execute()
# score = score.split("\n")[0]  # sometimes an explanation comes after score, ignore
# print(f"Scoring the dialogue as: {score}/10")
