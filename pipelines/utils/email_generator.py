import datetime

class EmailGenerator:
    """Composes and dispatches the final email."""

    def __init__(self, llm_client, prompt_manager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def generate_email_report(self, salesperson_name, client_recommendations):
        """Generates a daily email report for a salesperson."""
        today_date = datetime.date.today().strftime("%B %d, %Y")
        subject = f"Your Daily Client Research Recommendations: {today_date}"

        # Section 1: At-a-Glance Summary
        at_a_glance_summary = ""
        for client_id, recommendations in client_recommendations.items():
            at_a_glance_summary += f"\n### Client: {client_id}\n"
            for rec in recommendations:
                at_a_glance_summary += "- {} [Link] (Simulated Link)\n".format(rec.get("publication", {}).get("title", ""))

        # Section 2: Detailed Breakdown
        detailed_breakdown = ""
        for client_id, recommendations in client_recommendations.items():
            detailed_breakdown += f"\n## Detailed Breakdown for {client_id}\n"
            for rec in recommendations:
                detailed_breakdown += "\n### Publication: {} [Link] (Simulated Link)\n".format(rec.get("publication", {}).get("title", ""))
                detailed_breakdown += "Aggregate Score: {:.2f}\n".format(rec.get("aggregate_score", 0))
                detailed_breakdown += "Relevance Summary: {}\n".format(rec.get("relevance_summary", ""))
                detailed_breakdown += "\nSource of Recommendation:\n"
                detailed_breakdown += "| Source | Evidence | Score |\n"
                detailed_breakdown += "|---|---|---|\n"
                for source, data in rec.get("source_of_recommendation", {}).items():
                    detailed_breakdown += "| {} | {} | {} |\n".format(source, data.get("evidence", ""), data.get("score", 0))

        # Use LLM to compose the full email (simplified)
        email_prompt = self.prompt_manager.get_prompt("email_composition")
        full_email_content = self.llm_client.generate_completion(
            f"{email_prompt}\nSubject: {subject}\nSummary: {at_a_glance_summary}\nDetails: {detailed_breakdown}"
        )

        print(f"\n--- Email Report for {salesperson_name} ---")
        print(f"Subject: {subject}")
        print(full_email_content)
        print(f"\n--------------------------------------")

        return full_email_content


