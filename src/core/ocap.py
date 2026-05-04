import typing as t
from dataclasses import dataclass, field
import uuid

@dataclass(frozen=True)
class Capability:
    """An unforgeable token representing the authority to perform an action."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource: str = "none"
    action: str = "none"
    
    def __repr__(self) -> str:
        return f"Cap({self.resource}:{self.action} [{self.id[:8]}])"

class Membrane:
    """Wraps a resource or another membrane to restrict or log access (attenuation)."""
    def __init__(self, target: t.Any, allowed_actions: t.Set[str], label: str = "anonymous"):
        self._target = target
        self._allowed_actions = allowed_actions
        self._label = label

    def __getattr__(self, name: str) -> t.Any:
        if name not in self._allowed_actions:
            raise PermissionError(f"Action '{name}' is not allowed by membrane '{self._label}'")
        
        attr = getattr(self._target, name)
        if callable(attr):
            def wrapped(*args, **kwargs):
                return attr(*args, **kwargs)
            return wrapped
        return attr

class OCapManager:
    """Manages the distribution and verification of capabilities."""
    def __init__(self):
        self._registry: t.Dict[str, Capability] = {}

    def mint(self, resource: str, action: str) -> Capability:
        cap = Capability(resource=resource, action=action)
        self._registry[cap.id] = cap
        return cap

    def verify(self, cap: Capability, resource: str, action: str) -> bool:
        stored = self._registry.get(cap.id)
        if not stored:
            return False
        return stored.resource == resource and stored.action == action

    def attenuate(self, target: t.Any, allowed_actions: t.Set[str], label: str) -> Membrane:
        """Returns a membrane that restricts access to the target."""
        return Membrane(target, allowed_actions, label)
