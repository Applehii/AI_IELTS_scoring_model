from abc import ABC, abstractmethod

class BaseLLM(ABC):

    @abstractmethod
    def ask(self, messages, **kwargs) -> str:
        pass
