# %%
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


# %%
def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


# %%
def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | chat

    response = chain.invoke({"question": query, "docs": docs_page_content})
    response = response.content.replace("\n", "")
    return response, docs


# %%
# Best Dark Type Pokemon in Competitive play
video_url_1 = "https://youtu.be/6ldbXM75hfU?si=ihWHBlF44o4zhJt6"
db1 = create_db_from_youtube_video_url(video_url_1)

query1 = "What is the number 10 in the list of the best dark type pokemon in the video?"
response1, docs1 = get_response_from_query(db1, query1)
print("=== Video 1 ===")
print(textwrap.fill(response1, width=50))
docs1

# %%
# Investment Methods
video_url_2 = "https://youtu.be/Ay4fmZdZqJE?si=g4ZAFdo943iM7meh"
db2 = create_db_from_youtube_video_url(video_url_2)

query2 = "What is the investment method that consistently deliver returns over time?"
response2, docs2 = get_response_from_query(db2, query2)
print("=== Video 2 ===")
print(textwrap.fill(response2, width=50))
docs2


# %%
# Caro-Kann opening play as Black in Chess
video_url_3 = "https://www.youtube.com/watch?v=0p_881Nwoo4"
db3 = create_db_from_youtube_video_url(video_url_3)

query3 = "What are the pieces objectives when you play the Caro-Kann opening as black?"
response3, docs3 = get_response_from_query(db3, query3)
print("=== Video 3 ===")
print(textwrap.fill(response3, width=50))
docs3

# %%
