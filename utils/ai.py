from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate, StringPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories.file import FileChatMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)


def get_message_history(file_path: str) -> FileChatMessageHistory:
    file_path = os.path.join('data', file_path)
    return FileChatMessageHistory(file_path)


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Teddy, the task-manager AI assistant. Your primary task is to help the user manage their tasks."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}")
    ]
)

chain = chat_prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="user_input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="file_path",
            annotation=str,
            name="File Path",
            description="Unique identifier of the history",
            default="",
            is_shared=True
        )
    ]
)

output_parser = StrOutputParser()

full_chain = chain_with_history | output_parser

alt_chain = {'history' : lambda x: x['history'], 'user_input': lambda x: x['user_input']} | chat_prompt | llm | output_parser

def get_answer(user_input: str, history):
    return full_chain.invoke(
        {'user_input': user_input},
        config={"configurable": {"file_path": "history.txt"}}
    )

def get_answer_alt(user_input: str, history):
    intermediate_conversation = []
    for human, ai in history:
        intermediate_conversation.append(HumanMessage(content=human))
        intermediate_conversation.append(AIMessage(content=ai))
    return alt_chain.invoke({'history': intermediate_conversation, 'user_input': user_input})


# if __name__ == '__main__':
#     # output = chain_with_history.invoke(
#     #     {'user_input': "What is my name?"},
#     #     config={"configurable" : {"file_path": "history.txt"}}
#     # )

#     output = full_chain.invoke(
#         {'user_input': "How can you help me?"},
#         config={"configurable": {"file_path": "history.txt"}}
#     )

#     print(output)
