import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
    return text

pdf_text = extract_text_from_pdf('your_pdf_file_path.pdf')


##step 2:Entity Extraction with TransferBERT
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
ner_results = nlp(pdf_text)

#Step 3: Classify Entities with spaCy
#Classify the extracted entities using spaCy. You might need to train or fine-tune a model for specific entity types related to the hardware manual.

import spacy

nlp = spacy.load("en_core_web_sm")  # Use a suitable model

for entity in ner_results:
    doc = nlp(entity['word'])
    for ent in doc.ents:
        print(ent.text, ent.label_)


#Step 4: Create Knowledge Graph in Neo4j
#Finally, use the Neo4j Python driver to create nodes and relationships in the Neo4j database.

from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_entity(self, entity_name, entity_type):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_entity, entity_name, entity_type)

    @staticmethod
    def _create_and_return_entity(tx, entity_name, entity_type):
        query = (
            "CREATE (e:Entity {name: $entity_name, type: $entity_type}) "
            "RETURN e"
        )
        result = tx.run(query, entity_name=entity_name, entity_type=entity_type)
        return result.single()[0]

# Initialize and use the KnowledgeGraph class to connect and add data to Neo4j


### step.5 : Query to Knowledge Graph
from neo4j import GraphDatabase

class KnowledgeGraphQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_info(self, query):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_info, query)
            return result

    @staticmethod
    def _find_info(tx, query):
        query = f"MATCH (n:Entity) WHERE n.name CONTAINS '{query}' RETURN n"
        result = tx.run(query)
        return [record["n"] for record in result]

# Example usage
kg_query = KnowledgeGraphQuery("neo4j://localhost:7687", "username", "password")
kg_results = kg_query.get_info("troubleshooting issue")

#Step 6: Generate Few-Shot Prompt
#Use the information from the knowledge graph to generate a prompt for a language model. This prompt should encapsulate the context of the problem and any relevant details extracted from the knowledge graph.
def generate_prompt(user_query, kg_results):
    prompt = f"Troubleshooting Issue: {user_query}\n\nRelevant Information:\n"
    for result in kg_results:
        # Assuming each result has a 'description' property
        prompt += f"- {result['description']}\n"
    prompt += "\nHow can this issue be resolved?"
    return prompt

# Create a prompt based on user query and KG results
prompt = generate_prompt("Laptop not starting", kg_results)

#strp .7:llm for response generation
import openai

def get_model_response(prompt):
    openai.api_key = 'your-api-key'

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Get response from the model
model_response = get_model_response(prompt)


