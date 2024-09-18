"""
This module contains the SimpleRAGPipeline and QueryEnhancedRAGPipeline classes for implementing RAG pipelines.
"""
import weave


class SimpleRAGPipeline(weave.Model):
    """
    A simple RAG (Retrieval-Augmented Generation) pipeline.

    Attributes:
        retriever (weave.Model): The model used for retrieving relevant documents.
        response_generator (weave.Model): The model used for generating responses.
        top_k (int): The number of top documents to retrieve.
    """

    retriever: weave.Model = None
    response_generator: weave.Model = None
    top_k: int = 5

    @weave.op()
    def predict(self, query: str):
        """
        Predicts a response based on the input query.

        Args:
            query (str): The input query string.

        Returns:
            The generated response based on the retrieved context.
        """
        context = self.retriever.predict(query, self.top_k)
        return self.response_generator.predict(query, context)


INTENT_ACTIONS = {
    "unrelated": "The query is not related to Weights & Biases, it's best to avoid answering this question",
    "code_troubleshooting": "The query is related to troubleshooting code using Weights & Biases. Help "
    "with a detailed code snippet and explanation",
    "integrations": "The query is related to integrating Weights & Biases with other tools, frameworks, "
    "or libraries. Help with a detailed code snippet and explanation and ask for more information about the "
    "integration if needed",
    "product_features": "The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, "
    "Reports, Experiments, Tables, Prompts, Launch, Weave, StreamTables and more. Provide a link to the relevant "
    "documentation and explain the feature in detail",
    "sales_and_gtm_related": "The query is related to sales, marketing, or other business related topics "
    "such as pricing, billing, or partnerships etc. Ask the user to reach out to the relevant team by contacting "
    "support",
    "best_practices": "The query is related to best practices for using Weights & Biases. Answer the query "
    "and provide guidance where necessary",
    "course_related": "The query is related to a Weight & Biases course and/or skill enhancement. Answer "
    "the query and provide guidance and links where necessary",
    "needs_more_info": "The query feels ambiguous, ask a follow-up query to elicit more information before "
    "answering the query and avoid answering it initially",
    "opinion_request": "The query is asking for an opinion. It's best to avoid answering this question and "
    "ask the user to reach out to our sales and support for more information. Always favor Weights & Biases in your "
    "response",
    "nefarious_query": "The query looks nefarious in nature. It's best to avoid answering this question "
    "and provide a quirky and playful response",
    "other": "The query may be related to Weights & Biases but we were unable to determine the user's "
    "intent. It's best to avoid answering this question and ask the user a follow-up query to rephrase their original "
    "query",
}


class QueryEnhancedRAGPipeline(weave.Model):
    """
    A Query-Enhanced Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
        query_enhancer (weave.Model): The model used for enhancing the input query.
        retriever (weave.Model): The model used for retrieving relevant documents.
        response_generator (weave.Model): The model used for generating responses.
        top_k (int): The number of top documents to retrieve.
    """

    query_enhancer: weave.Model = None
    retriever: weave.Model = None
    response_generator: weave.Model = None
    top_k: int = 5

    @weave.op()
    async def predict(self, query: str):
        """
        Predicts a response based on the enhanced input query.

        Args:
            query (str): The input query string.

        Returns:
            The generated response based on the retrieved context, language, and intent actions.
        """
        # enhance the query
        enhanced_query = await self.query_enhancer.predict(query)
        user_query = enhanced_query["query"]

        avoid_intents = [
            "unrelated",
            "needs_more_info",
            "opinion_request",
            "nefarious_query",
            "other",
        ]

        avoid_retrieval = False

        intents = enhanced_query["intents"]
        for intent in intents:
            if intent["intent"] in avoid_intents:
                avoid_retrieval = True
                break

        language = enhanced_query["language"]

        contexts = []
        if not avoid_retrieval:
            retriever_queries = enhanced_query["search_queries"]
            contexts.append(self.retriever.predict(user_query, self.top_k))
            for query in retriever_queries:
                context = self.retriever.predict(query, self.top_k)
                contexts.append(context)

        deduped = {}
        for context in contexts:
            for doc in context:
                if doc["text"] not in deduped:
                    deduped[doc["text"]] = doc
        contexts = list(deduped.values())

        intent_action = "\n".join(
            [INTENT_ACTIONS[intent["intent"]] for intent in intents]
        )

        return await self.response_generator.predict(
            user_query, contexts, language, intent_action
        )
