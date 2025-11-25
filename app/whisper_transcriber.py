import whisper

class Transcriber:
    def __init__(self, model_name="medium"):
        """
        model_name: tiny, base, small, medium, large
        """
        self.model = whisper.load_model(model_name)

    def transcribe(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result.get("text", "").strip()
