from langchain_core.tools import tool

@tool 
def followup_handler(query:str):
    '''
    -  This tool is designed to resturcture the vague or underspecified follow-up query.
    If the query is identified as a follow-up,the tool evaluates whether it is vague or 
    underspecified. In cases where the query lacks clarity, context, or explicit details, 
    the tool restructures and rephrases the query into a clearer and more explicit form 
    that is easier for the LLM to understand and respond to accurately.

    Key responsibilities:
    - Detect if the query is a follow-up question.
    - Identify vague, incomplete, or ambiguous queries.
    - Restructure vague follow-up queries into well-formed, self-contained and detailed queries like shown in example.

    Intended use:
    - To improve conversational continuity and reduce ambiguity.
    - To ensure the LLM receives clear, structured inputs, leading to 
      more accurate and context-aware responses

    Example:
    ---
    - User: "did you do the comedy factory project"
    - Assistant: "Yes"
    - User: "What was the Techstack"
    - Tool → Restructured: "What was the Techstack of comedy factory project"
    '''