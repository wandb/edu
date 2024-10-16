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
    "financial_performance": "The query is related to financial performance such as revenue, profit, margins, or overall financial health. Provide detailed analysis based on the available financial reports.",
    "operational_metrics": "The query is about specific business metrics, KPIs, or operational performance. Analyze the relevant metrics from the financial reports.",
    "market_analysis": "The query is related to market share, competition, or industry trends. Provide insights based on the company's disclosures and market information in the reports.",
    "risk_assessment": "The query is about potential risks, legal issues, or uncertainties facing the company. Analyze the risk factors and management's discussion in the reports.",
    "strategic_initiatives": "The query is about company strategy, new products/services, or future plans. Provide information based on management's strategic discussions in the reports.",
    "accounting_practices": "The query is about specific accounting methods, policies, or financial reporting practices. Explain the relevant accounting principles and their application.",
    "management_insights": "The query is related to management commentary, guidance, or leadership decisions. Analyze the management's discussion and analysis sections of the reports.",
    "capital_structure": "The query is about debt, equity, capital allocation, or financing activities. Provide analysis based on the balance sheet and cash flow statements.",
    "segment_analysis": "The query is about performance or metrics of specific business segments or divisions. Analyze the segment reporting in the financial statements.",
    "comparative_analysis": "The query is comparing current results to past periods or to other companies. Provide a comparative analysis using the available financial data.",
    "unrelated": "The query is not related to financial analysis or SEC filings. It's best to avoid answering this question and ask for a finance-related query.",
    "needs_more_info": "The query is ambiguous or lacks context. Ask a follow-up question to elicit more specific information about the financial analysis needed.",
    "opinion_request": "The query is asking for a subjective opinion rather than factual analysis. Clarify that as an AI, you provide objective analysis based on financial reports, not personal opinions.",
    "nefarious_query": "The query appears to have potentially unethical or malicious intent. Avoid answering and suggest focusing on legitimate financial analysis questions.",
    "other": "The query may be related to financial analysis, but its intent is unclear. Ask the user to rephrase their question, focusing specifically on aspects of financial reports or SEC filings."
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
