from agent.tools import search_docs

# We are using 'search_docs' because that is the name 
# defined with the @tool decorator in your tools.py
response = search_docs.invoke("What is this project about?")

print("\n--- AGENT RETRIEVED DATA ---")
print(response)
print("----------------------------")