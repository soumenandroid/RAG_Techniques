import os
import tempfile
from langchain_core.runnables import(
    RunnableParallel,
    RunnablePassthrough
)
from langchain_community.graphs import Neo4jGraph
from docx import Document
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.output_parsers import StrOutputParser

class RAG_Graph:
    
    load_dotenv(find_dotenv())

    default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 100"

    def __init__(self):
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"]= "password"
        self.graph = Neo4jGraph()
        self.llm = ChatGroq(temperature = 0.5,groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama3-70b-8192")
    
    def create_graph(self,docs,TMP_DIR):
        for source_docs in docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(),suffix='.docx') as temp_file:
                temp_file.write(source_docs.read())
        
        loader = DirectoryLoader(TMP_DIR.as_posix(),glob='**/*docx',show_progress=True)
        self.document = loader.load()

        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(self.document)
        print(texts)

        llm_transformer = LLMGraphTransformer(llm=self.llm)
        #Extract graph data
        graph_documents = llm_transformer.convert_to_graph_documents(texts)
        vector_index_config = {"vector": {"type": "embedding", "dimension": 768}}

        #Store to Neo4J
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True,
            
        )

    def create_vector_index(self):
        model_name='sentence-transformers/all-mpnet-base-v2'
       # model_name='sentence-transformers/all-MiniLM-L6-v2'
        ''' self.vector_index = Neo4jVector.from_existing_index(
            HuggingFaceEmbeddings(model_name =model_name,
                                         model_kwargs={'device':'cpu'}),
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            index_name='entity',
        ) '''
        #self.vector_index = 'entity'
        self.vector_index = Neo4jVector.from_existing_graph(
            HuggingFaceEmbeddings(model_name =model_name,
                                         model_kwargs={'device':'cpu'}),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
    
    def prepare_chat_template(self):
        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are extracting fields and business rules from the text"
                    ),
                    (
                        "human",
                        "Use this given format to extract the information from the following"
                        "input: {question}",

                    ),
                ]
        )
        self.entity_chain = prompt | self.llm.with_structured_output(Entities)

    def retriever(self,question: str):
        print(f"Search query: {question}")
        structure_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        final_data = f"""Structured data:
        {structure_data}
        Unstructured data:
        {"#Document ".join(unstructured_data)}
        """
        return final_data

    def structured_retriever(self, question: str) -> str:
        result = ""
        entities = self.entity_chain.invoke({"question": question})
       
        #This will return a self query for all nodes and neighbours
        for entity in entities.names:
            response = self.graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": self.generate_full_text_query(entity)},
            )
            # Joins the results from the query into a single string.
            result += "\n".join([el['output'] for el in response])
        return result

    @staticmethod
    def generate_full_text_query(input: str) -> str:
        full_text_query = ""
        words = [e1 for e1 in remove_lucene_chars(input).split() if e1]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()


    def ask_question_chain(self, query):
        self.create_vector_index()
        self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        
        self.prepare_chat_template()

        template = """Answer the question based only on the following context
        {context}
        
        Question: {question}
        User natrual language and be concise.
        Answer: """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnableParallel(
                {
                    "context" : self.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        result = chain.invoke(query)
        return result

    def retriever1(self,prompt1):
        # Retriever

        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        # Extract entities from text
        class Entities(BaseModel):
            """Identifying information about entities."""

            names: List[str] = Field(
                ...,
                description="All the person, organization, or business entities that "
                "appear in the text",
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        entity_chain = prompt | self.llm.with_structured_output(Entities)
        result = entity_chain.invoke(prompt1)
        print(result)


class Entities(BaseModel):
    """ Identify information about entities"""
    names: List[str] = Field(
        ...,
        description = "All the fields, or business rules that appear in the text"
    )
