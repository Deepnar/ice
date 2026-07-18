"""Core helpers for the mini fixture package."""


def greet(name):
    """Return a friendly greeting.

    The second docstring line must never be stored (first line only).
    """
    return f"hello {name}"


def helper():
    """A uniquely named helper (heuristic call-edge target)."""
    return 42


class Animal:
    """Base animal."""

    def speak(self):
        """Make a generic noise."""
        return "..."


class Dog(Animal):
    """A dog that greets."""

    def speak(self):
        """Bark by delegating to greet."""
        return greet("dog") + self.tail_wag()

    def tail_wag(self):
        """Wag."""
        return "!"
