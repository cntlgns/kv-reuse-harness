
class MaxRecursionsReachedException(Exception):
    """Exception raised when the agent event-loop reaches its maximum set recursion limit.

    This is raised as a protective mechanism for long-running conversation b/w agent and environment
    Raise this exception to reset the recursive nature of tool-calling
    """

    def __init__(self, message: str):
        """Initialize the exception with an error message and the incomplete message object.

        Args:
            message: The error message describing the token limit issue
        """
        super().__init__(message)