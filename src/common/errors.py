class PromptNotFoundError(Exception):
    pass


class PromptDataError(Exception):
    pass


class LLMUpstreamError(Exception):
    def __init__(self, status, message, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


class PersistenceError(Exception):
    pass
