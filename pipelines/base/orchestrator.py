from core_pipeline.profiling import Profiling
from core_pipeline.recommendation.retriever import Retriever
from core_pipeline.recommendation.ranker import Ranker
from llm_services.llm_client import LLMClient
from llm_services.prompt_manager import PromptManager

class Orchestrator:
    """Runs the full recommendation pipeline."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()
        self.profiling = Profiling()
        self.retriever = Retriever(self.llm_client)
        self.ranker = Ranker(self.llm_client, self.prompt_manager)

    def run_pipeline(self, client_id, all_publications):
        """Executes the recommendation pipeline for a given client."""
        print(f"Running pipeline for client: {client_id}")

        # 1. Build Client Profile (simplified)
        client_profile = self.profiling.build_client_profile(client_id)
        print(f"Client Profile: {client_profile}")

        # 2. Build Publication Profiles and add to retriever (simplified)
        for pub_id, pub_data in all_publications.items():
            pub_profile = self.profiling.build_publication_profile(pub_id, pub_data)
            self.retriever.add_publication_embedding(pub_id, pub_profile["embedding"])

        # 3. Retrieve Candidates
        # For simplicity, using a dummy client embedding for retrieval
        client_embedding_for_retrieval = self.llm_client.generate_embedding(str(client_profile))
        candidate_publication_ids = self.retriever.retrieve_candidates(client_embedding_for_retrieval)
        candidate_publications = [all_publications[pub_id] for pub_id in candidate_publication_ids]
        print(f"Candidate Publications: {candidate_publications}")

        # 4. Score and Rank
        ranked_recommendations = self.ranker.score_and_rank(client_profile, candidate_publications)
        print(f"Ranked Recommendations: {ranked_recommendations}")

        return ranked_recommendations


