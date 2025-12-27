from app.rag_manager import RAGManager
from app.pipeline.utils import (
    extract_rubric,
    extract_summary,
    trim_context
)
from app.pipeline.writing_task1 import WritingTask1Pipeline
from app.pipeline.writing_task2 import WritingTask2Pipeline


class WritingPipeline:
    def __init__(self):
        self.rag = RAGManager()
        self.task1_pipeline = WritingTask1Pipeline()
        self.task2_pipeline = WritingTask2Pipeline()

    # --------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------
    def score_writing(
        self,
        question: str,
        answer: str,
        chart_path: str | None = None
    ):
        if chart_path:
            return self.task1_pipeline.score(
                question=question,
                answer=answer,
                chart_path=chart_path
            )

        context = self.build_context(question)

        return self.task2_pipeline.score(
            question=question,
            answer=answer,
            context=context
        )
